import time
import numpy as np

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

    # Try to reduce rank to 21
    rank = U.shape[1]
    found = False
    best_U, best_V, best_W = U, V, W
    
    # Pre-compute A matrix structure for each possible removal?
    # No, just do it in the loop.
    
    while time.time() - START_TIME < TIME_BUDGET - 5:
        # Randomly pick a column to remove
        r_to_remove = np.random.randint(rank)
        mask = [i for i in range(rank) if i != r_to_remove]
        U_r = U[:, mask]
        V_r = V[:, mask]
        nr = len(mask)
        
        # Build linear system
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
                # Check if U_r and V_r are also half-integers (they are, from rank 22)
                # But we should verify the whole thing
                if _verify(U_r, V_r, W_rounded, T):
                    best_U, best_V, best_W = U_r, V_r, W_rounded
                    found = True
                    break
        except np.linalg.LinAlgError:
            continue

    if found:
        U, V, W = best_U, best_V, best_W

    # Round final result to clean values
    U = np.round(U, 10)
    V = np.round(V, 10)
    W = np.round(W, 10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
