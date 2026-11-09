import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_BUDGET = 140.0

def _tensor_index(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0.0)

def _build_rank22_decomposition(n, m, p):
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        col = col_start
        # M1
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M2
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        # M3
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M4
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        # M5
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        # M6
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M7
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    c1 = {'00': 0, '01': 2, '10': 1, '11': 3}
    col = _add_strassen_block(a1, b1, c1, 0)

    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    col = _add_strassen_block(a1, b2, c2, col)

    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    return U, V, W

def _try_reduce_rank(U, V, W, T, n, m, p, target_rank, time_limit=10.0):
    """Try to reduce rank from current to target_rank by removing columns."""
    start = time.time()
    current_rank = U.shape[1]
    cols_to_remove = current_rank - target_rank
    
    if cols_to_remove <= 0:
        return U, V, W
    
    # Pre-compute the linear system structure for each column removal pattern
    # For efficiency, we'll try systematic removals first, then random
    
    # Systematic: try all single column removals first
    for r_to_remove in range(current_rank):
        if time.time() - start > time_limit:
            break
        mask = [i for i in range(current_rank) if i != r_to_remove]
        if len(mask) < target_rank:
            continue
        U_r = U[:, mask]
        V_r = V[:, mask]
        nr = len(mask)
        
        # Build linear system for W
        A_mat = np.zeros((n * m * p, p * nr))
        b_vec = T.flatten()
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(nr):
                        A_mat[idx, k * nr + r_idx] = U_r[i, r_idx] * V_r[j, r_idx]
                    idx += 1
        
        try:
            W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_sol = W_sol.reshape(p, nr)
            
            # Round to half-integers
            W_rounded = np.round(W_sol * 2) / 2.0
            
            # Verify
            reconstructed = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_rounded)
            if np.allclose(reconstructed, T, atol=1e-9, rtol=0.0):
                return U_r, V_r, W_rounded
        except np.linalg.LinAlgError:
            continue
    
    # If we need to remove more than 1 column, try combinations
    if cols_to_remove > 1:
        # Try all combinations of cols_to_remove columns to remove
        from itertools import combinations
        for combo in combinations(range(current_rank), cols_to_remove):
            if time.time() - start > time_limit:
                break
            mask = [i for i in range(current_rank) if i not in combo]
            U_r = U[:, mask]
            V_r = V[:, mask]
            nr = len(mask)
            
            A_mat = np.zeros((n * m * p, p * nr))
            b_vec = T.flatten()
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(nr):
                            A_mat[idx, k * nr + r_idx] = U_r[i, r_idx] * V_r[j, r_idx]
                        idx += 1
            
            try:
                W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_sol = W_sol.reshape(p, nr)
                
                W_rounded = np.round(W_sol * 2) / 2.0
                
                reconstructed = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_rounded)
                if np.allclose(reconstructed, T, atol=1e-9, rtol=0.0):
                    return U_r, V_r, W_rounded
            except np.linalg.LinAlgError:
                continue
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _tensor_index(n, m, p)
    U, V, W = _build_rank22_decomposition(n, m, p)
    
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank systematically
    # First try rank 21
    best_U, best_V, best_W = U, V, W
    current_rank = 22
    
    # Try rank 21 with a reasonable time budget
    time_for_21 = min(30.0, TIME_BUDGET - (time.time() - START_TIME) - 10)
    if time_for_21 > 2:
        U21, V21, W21 = _try_reduce_rank(U, V, W, T, n, m, p, 21, time_limit=time_for_21)
        if U21.shape[1] < current_rank:
            best_U, best_V, best_W = U21, V21, W21
            current_rank = U21.shape[1]
    
    # Try rank 20 if we have time
    time_for_20 = min(20.0, TIME_BUDGET - (time.time() - START_TIME) - 8)
    if current_rank > 20 and time_for_20 > 2:
        U20, V20, W20 = _try_reduce_rank(best_U, best_V, best_W, T, n, m, p, 20, time_limit=time_for_20)
        if U20.shape[1] < current_rank:
            best_U, best_V, best_W = U20, V20, W20
            current_rank = U20.shape[1]
    
    # Use remaining time for random search at current best rank - 1
    remaining = TIME_BUDGET - (time.time() - START_TIME) - 5
    if remaining > 5 and current_rank > 20:
        target = current_rank - 1
        U_rand, V_rand, W_rand = _try_reduce_rank(best_U, best_V, best_W, T, n, m, p, target, time_limit=remaining)
        if U_rand.shape[1] < current_rank:
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            current_rank = U_rand.shape[1]

    U, V, W = best_U, best_V, best_W

    # Round final result to clean values
    U = np.round(U, 10)
    V = np.round(V, 10)
    W = np.round(W, 10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
