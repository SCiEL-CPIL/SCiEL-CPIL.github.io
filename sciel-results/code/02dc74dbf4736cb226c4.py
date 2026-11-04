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


def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Greedily remove columns (singles, pairs, triples) that are redundant.
    Uses incremental tensor updates for speed. Multiple restarts with
    randomized order to escape local minima.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    target = np.ones((n, m, p))
    tol = 1e-9
    
    # Precompute per-column tensor contributions
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    best_keep = np.ones(rank, dtype=bool)
    best_rank = rank
    
    def _greedy_pass(keep, T_current, deadline):
        """Run greedy single/pair/triple removal until fixed point."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Singles
            for r in range(rank):
                if not keep[r]:
                    continue
                T_new = T_current - col_contrib[r]
                if np.allclose(T_new, target, atol=tol):
                    keep[r] = False
                    T_current = T_new
                    changed = True
                    break
            if changed:
                continue
            # Pairs
            kept_indices = [r for r in range(rank) if keep[r]]
            found = False
            for i in range(len(kept_indices)):
                if time.time() > deadline:
                    break
                r1 = kept_indices[i]
                if not keep[r1]:
                    continue
                for j in range(i + 1, len(kept_indices)):
                    r2 = kept_indices[j]
                    if not keep[r2]:
                        continue
                    T_new = T_current - col_contrib[r1] - col_contrib[r2]
                    if np.allclose(T_new, target, atol=tol):
                        keep[r1] = False
                        keep[r2] = False
                        T_current = T_new
                        found = True
                        break
                if found:
                    break
            if found:
                continue
            # Triples
            kept_indices = [r for r in range(rank) if keep[r]]
            found = False
            for i in range(len(kept_indices)):
                if time.time() > deadline:
                    break
                r1 = kept_indices[i]
                if not keep[r1]:
                    continue
                for j in range(i + 1, len(kept_indices)):
                    if time.time() > deadline:
                        break
                    r2 = kept_indices[j]
                    if not keep[r2]:
                        continue
                    for k in range(j + 1, len(kept_indices)):
                        r3 = kept_indices[k]
                        if not keep[r3]:
                            continue
                        T_new = T_current - col_contrib[r1] - col_contrib[r2] - col_contrib[r3]
                        if np.allclose(T_new, target, atol=tol):
                            keep[r1] = False
                            keep[r2] = False
                            keep[r3] = False
                            T_current = T_new
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return T_current
    
    # First pass: deterministic order
    keep = np.ones(rank, dtype=bool)
    T_current = col_contrib.sum(axis=0)
    T_current = _greedy_pass(keep, T_current, deadline)
    cur_rank = int(np.sum(keep))
    if cur_rank < best_rank:
        best_rank = cur_rank
        best_keep = keep.copy()
    
    # Additional randomized restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        # Randomize the order of consideration by permuting column indices
        keep = np.ones(rank, dtype=bool)
        T_current = col_contrib.sum(axis=0)
        # Use a shuffled order for the greedy pass
        perm = rng.permutation(rank)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Singles in random order
            for r in perm:
                if not keep[r]:
                    continue
                T_new = T_current - col_contrib[r]
                if np.allclose(T_new, target, atol=tol):
                    keep[r] = False
                    T_current = T_new
                    changed = True
                    break
            if changed:
                continue
            # Pairs in random-ish order
            kept_indices = [r for r in perm if keep[r]]
            found = False
            for i in range(len(kept_indices)):
                if time.time() > deadline:
                    break
                r1 = kept_indices[i]
                if not keep[r1]:
                    continue
                for j in range(i + 1, len(kept_indices)):
                    r2 = kept_indices[j]
                    if not keep[r2]:
                        continue
                    T_new = T_current - col_contrib[r1] - col_contrib[r2]
                    if np.allclose(T_new, target, atol=tol):
                        keep[r1] = False
                        keep[r2] = False
                        T_current = T_new
                        found = True
                        break
                if found:
                    break
            if found:
                continue
            # Triples
            kept_indices = [r for r in perm if keep[r]]
            found = False
            for i in range(len(kept_indices)):
                if time.time() > deadline:
                    break
                r1 = kept_indices[i]
                if not keep[r1]:
                    continue
                for j in range(i + 1, len(kept_indices)):
                    if time.time() > deadline:
                        break
                    r2 = kept_indices[j]
                    if not keep[r2]:
                        continue
                    for k in range(j + 1, len(kept_indices)):
                        r3 = kept_indices[k]
                        if not keep[r3]:
                            continue
                        T_new = T_current - col_contrib[r1] - col_contrib[r2] - col_contrib[r3]
                        if np.allclose(T_new, target, atol=tol):
                            keep[r1] = False
                            keep[r2] = False
                            keep[r3] = False
                            T_current = T_new
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        cur_rank = int(np.sum(keep))
        if cur_rank < best_rank:
            best_rank = cur_rank
            best_keep = keep.copy()
    
    if best_rank >= rank:
        return U, V, W, rank
    
    U_new = U_arr[:, best_keep]
    V_new = V_arr[:, best_keep]
    W_new = W_arr[:, best_keep]
    
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, best_rank
    else:
        return U, V, W, rank


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    r = 0
    
    # Strassen coefficients
    strassen_u = {
        0: [(0, 1), (4, 1)],
        1: [(3, 1), (4, 1)],
        2: [(0, 1)],
        3: [(4, 1)],
        4: [(0, 1), (1, 1)],
        5: [(3, 1), (0, -1)],
        6: [(1, 1), (4, -1)],
    }
    
    strassen_v = {
        0: [(0, 1), (11, 1)],
        1: [(0, 1)],
        2: [(1, 1), (11, -1)],
        3: [(10, 1), (0, -1)],
        4: [(11, 1)],
        5: [(0, 1), (1, 1)],
        6: [(10, 1), (11, 1)],
    }
    
    strassen_w = {
        0: [(0, 1), (3, 1)],
        1: [(1, 1), (3, -1)],
        2: [(2, 1), (3, 1)],
        3: [(0, 1), (1, 1)],
        4: [(0, -1), (2, 1)],
        5: [(3, 1)],
        6: [(0, 1)],
    }
    
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        
        for s in range(7):
            for (u_row, coeff) in strassen_u[s]:
                U[u_row][r] += coeff
            
            for (v_row_offset, coeff) in strassen_v[s]:
                if v_row_offset == 0:
                    v_row = col0
                elif v_row_offset == 1:
                    v_row = col1
                elif v_row_offset == 10:
                    v_row = 10 + col0
                elif v_row_offset == 11:
                    v_row = 10 + col1
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
    
    # Schoolbook for A[:,2] * B[2,:]
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[2 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s cap

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)
    else:
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
