"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _tensor_contribution(n, m, p, U_arr, V_arr, W_arr, keep_mask):
    """Compute the tensor sum over kept columns."""
    U3 = U_arr[:, keep_mask].reshape(n, m, -1)
    V3 = V_arr[:, keep_mask].reshape(m, p, -1)
    W3 = W_arr[:, keep_mask].reshape(p, n, -1)
    if U3.shape[2] == 0:
        return np.zeros((n, m, p))
    return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)


def _build_strassen(n, m, p, partition='21'):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    partition: '21' means rows 0,1 use Strassen, row 2 uses schoolbook
               '12' means row 0 uses schoolbook, rows 1,2 use Strassen
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    r = 0
    
    # Strassen coefficients for 2x2x2
    strassen_u = {
        0: [(0, 1), (1, 1)],
        1: [(1, 1), (1, 1)],
        2: [(0, 1)],
        3: [(1, 1)],
        4: [(0, 1), (0, 1)],
        5: [(1, 1), (0, -1)],
        6: [(0, 1), (1, -1)],
    }
    
    strassen_v = {
        0: [(0, 1), (1, 1)],
        1: [(0, 1)],
        2: [(1, 1), (0, -1)],
        3: [(0, 1), (1, -1)],
        4: [(1, 1)],
        5: [(0, 1), (1, 1)],
        6: [(0, 1), (1, 1)],
    }
    
    strassen_w = {
        0: [(0, 1), (1, 1)],
        1: [(1, 1), (0, -1)],
        2: [(1, 1), (1, 1)],
        3: [(0, 1), (0, 1)],
        4: [(0, -1), (1, 1)],
        5: [(1, 1)],
        6: [(0, 1)],
    }
    
    if partition == '21':
        strassen_rows = [0, 1]
        schoolbook_row = 2
    else:
        strassen_rows = [1, 2]
        schoolbook_row = 0
    
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        
        for s in range(7):
            for (u_row, coeff) in strassen_u[s]:
                U[strassen_rows[u_row] * m + strassen_rows[u_row]][r] += coeff
            
            for (v_row_offset, coeff) in strassen_v[s]:
                if v_row_offset == 0:
                    v_row = strassen_rows[0] * p + col0
                elif v_row_offset == 1:
                    v_row = strassen_rows[1] * p + col1
                else:
                    raise ValueError(f"Unexpected v_row_offset {v_row_offset}")
                V[v_row][r] += coeff
            
            for (w_row_offset, coeff) in strassen_w[s]:
                if w_row_offset == 0:
                    w_row = col0 * 2 + 0
                elif w_row_offset == 1:
                    w_row = col0 * 2 + 1
                elif w_row_offset == 2:
                    w_row = col1 * 2 + 0
                elif w_row_offset == 3:
                    w_row = col1 * 2 + 1
                else:
                    raise ValueError(f"Unexpected w_row_offset {w_row_offset}")
                W[w_row][r] += coeff
            
            r += 1
    
    # Schoolbook for the remaining row
    for i in range(2):
        for k in range(10):
            U[i * m + schoolbook_row][r] = 1
            V[schoolbook_row * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def _try_reduce_rank(n, m, p, U, V, W, deadline, aggressive=False):
    """
    Greedily remove columns that are redundant. If aggressive=True, also try
    removing pairs of columns and applying random basis changes.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    keep = np.ones(rank, dtype=bool)
    
    target = np.ones((n, m, p))
    
    # Precompute per-column tensor contributions
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    # Current total tensor
    T_current = col_contrib.sum(axis=0)
    
    # Iteratively try to remove columns
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            T_new = T_current - col_contrib[r]
            if np.allclose(T_new, target, atol=1e-9):
                keep[r] = False
                T_current = T_new
                changed = True
                break
    
    if aggressive and time.time() < deadline:
        # Try removing pairs
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for r1 in range(rank):
                if not keep[r1]:
                    continue
                for r2 in range(r1 + 1, rank):
                    if not keep[r2]:
                        continue
                    T_new = T_current - col_contrib[r1] - col_contrib[r2]
                    if np.allclose(T_new, target, atol=1e-9):
                        keep[r1] = False
                        keep[r2] = False
                        T_current = T_new
                        changed = True
                        break
                if changed:
                    break
    
    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank
    
    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]
    
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, new_rank
    else:
        return U, V, W, rank


def _try_basis_change(n, m, p, U, V, W, deadline, num_trials=100):
    """
    Try random integer basis changes to expose more redundancies.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    rng = np.random.RandomState(42)
    
    for trial in range(num_trials):
        if time.time() >= deadline:
            break
        
        # Generate a random integer matrix for basis change
        # Keep it simple: use small integers
        scale = 2
        Q = rng.randint(-scale, scale + 1, size=(rank, rank))
        
        # Check if Q is invertible
        det = np.linalg.det(Q)
        if abs(det) < 1e-6:
            continue
        
        # Apply basis change: new columns = old columns * Q
        # U_new = U * Q, V_new = V * Q, W_new = W * Q
        U_new = U_arr @ Q
        V_new = V_arr @ Q
        W_new = W_arr @ Q
        
        # Check if entries are close to integers
        if not np.allclose(U_new, np.round(U_new), atol=1e-6):
            continue
        if not np.allclose(V_new, np.round(V_new), atol=1e-6):
            continue
        if not np.allclose(W_new, np.round(W_new), atol=1e-6):
            continue
        
        U_list = np.round(U_new).tolist()
        V_list = np.round(V_new).tolist()
        W_list = np.round(W_new).tolist()
        
        if not _validate(n, m, p, U_list, V_list, W_list):
            continue
        
        # Try to reduce rank
        reduced_U, reduced_V, reduced_W, reduced_rank = _try_reduce_rank(
            n, m, p, U_list, V_list, W_list, deadline, aggressive=True
        )
        
        if reduced_rank < best_rank:
            best_U, best_V, best_W, best_rank = reduced_U, reduced_V, reduced_W, reduced_rank
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 120s cap

    n, m, p = DIMENSIONS

    # Try both partitions
    U1, V1, W1, rank1 = _build_strassen(n, m, p, partition='21')
    U2, V2, W2, rank2 = _build_strassen(n, m, p, partition='12')
    
    # Validate both
    valid1 = _validate(n, m, p, U1, V1, W1)
    valid2 = _validate(n, m, p, U2, V2, W2)
    
    if not valid1 and not valid2:
        U, V, W, rank = _build_trivial(n, m, p)
    elif valid1 and not valid2:
        U, V, W, rank = U1, V1, W1, rank1
    elif not valid1 and valid2:
        U, V, W, rank = U2, V2, W2, rank2
    else:
        # Both valid, use the one that reduces better
        U1r, V1r, W1r, rank1r = _try_reduce_rank(n, m, p, U1, V1, W1, deadline)
        U2r, V2r, W2r, rank2r = _try_reduce_rank(n, m, p, U2, V2, W2, deadline)
        
        if rank1r <= rank2r:
            U, V, W, rank = U1r, V1r, W1r, rank1r
        else:
            U, V, W, rank = U2r, V2r, W2r, rank2r
    
    # Try basis change if we have time
    if time.time() < deadline - 30 and rank > 50:
        U, V, W, rank = _try_basis_change(n, m, p, U, V, W, deadline, num_trials=50)
    
    # Final validation
    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
