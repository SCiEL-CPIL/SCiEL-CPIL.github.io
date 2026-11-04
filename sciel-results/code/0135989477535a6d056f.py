"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


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


def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Greedily remove columns that are redundant in the sense that the remaining
    columns still produce the correct tensor. Uses incremental tensor updates
    for speed.
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
    
    T_current = col_contrib.sum(axis=0)
    
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
    
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[2 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def _try_optimize_rank(n, m, p, start_rank, deadline):
    """
    Try to find a decomposition of a given rank using scipy optimization.
    Variables are constrained to multiples of 0.5.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = n, m, p
    num_u = n * m
    num_v = m * p
    num_w = p * n
    total_vars = (num_u + num_v + num_w) * start_rank
    
    # Objective: sum of squared errors in the tensor
    def _objective(x):
        U_flat = x[:num_u * start_rank]
        V_flat = x[num_u * start_rank : num_u * start_rank + num_v * start_rank]
        W_flat = x[num_u * start_rank + num_v * start_rank :]
        
        U_arr = U_flat.reshape(n, m, start_rank)
        V_arr = V_flat.reshape(m, p, start_rank)
        W_arr = W_flat.reshape(p, n, start_rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr)
        target = np.ones((n, m, p))
        return np.sum((T - target) ** 2)
    
    # Start from the rank-55 solution, zero-padded to start_rank if start_rank > 55,
    # or truncated if start_rank < 55 (won't work, so we only try start_rank <= 55)
    if start_rank > 55:
        return None
    
    # Get initial rank-55 solution
    U55, V55, W55, _ = _build_strassen(n, m, p)
    U55_arr = np.array(U55, dtype=float)
    V55_arr = np.array(V55, dtype=float)
    W55_arr = np.array(W55, dtype=float)
    
    # Create initial guess: use first start_rank columns of the rank-55 solution
    x0 = np.zeros(total_vars)
    x0[:num_u * start_rank] = U55_arr[:, :start_rank].flatten()
    x0[num_u * start_rank : num_u * start_rank + num_v * start_rank] = V55_arr[:, :start_rank].flatten()
    x0[num_u * start_rank + num_v * start_rank :] = W55_arr[:, :start_rank].flatten()
    
    # Try optimization
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    if result.fun > 1e-8:
        return None
    
    # Extract and round to multiples of 0.5
    x_opt = result.x
    U_opt = x_opt[:num_u * start_rank].reshape(n, m, start_rank)
    V_opt = x_opt[num_u * start_rank : num_u * start_rank + num_v * start_rank].reshape(m, p, start_rank)
    W_opt = x_opt[num_u * start_rank + num_v * start_rank :].reshape(p, n, start_rank)
    
    # Round to nearest 0.5
    U_rounded = np.round(U_opt * 2) / 2
    V_rounded = np.round(V_opt * 2) / 2
    W_rounded = np.round(W_opt * 2) / 2
    
    # Convert to lists
    U_list = U_rounded.reshape(-1, start_rank).tolist()
    V_list = V_rounded.reshape(-1, start_rank).tolist()
    W_list = W_rounded.reshape(-1, start_rank).tolist()
    
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, start_rank
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        # Fallback to trivial rank 60
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
    else:
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)
        
        # Try to find even lower rank using optimization
        for target_rank in range(rank - 1, max(rank - 10, 29), -1):
            if time.time() > deadline - 5:
                break
            result = _try_optimize_rank(n, m, p, target_rank, deadline)
            if result is not None:
                U, V, W, rank = result
                break

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
