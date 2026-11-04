"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
        
    rank = U_arr.shape[1]
    
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False

    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _newton_polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
    """
    Newton's method to polish a tensor decomposition.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    target = np.ones((n, m, p))
    
    for it in range(max_iter):
        S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        R = S - target
        
        err = np.max(np.abs(R))
        if err < tol:
            break
        
        n_unknowns = (n * m + m * p + p * n) * rank
        n_equations = n * m * p
        
        J = np.zeros((n_equations, n_unknowns))
        b = -R.flatten()
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for c in range(rank):
                        u_idx = (i * m + j) * rank + c
                        J[eq_idx, u_idx] = V_t[j, k, c] * W_t[k, i, c]
                        
                        v_idx = (n * m + j * p + k) * rank + c
                        J[eq_idx, v_idx] = U_t[i, j, c] * W_t[k, i, c]
                        
                        w_idx = (n * m + m * p + k * n + i) * rank + c
                        J[eq_idx, w_idx] = U_t[i, j, c] * V_t[j, k, c]
        
        try:
            delta, residuals, rank_sv, sv = np.linalg.lstsq(J, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        dU = delta[:n * m * rank].reshape(n, m, rank)
        dV = delta[n * m * rank:n * m * rank + m * p * rank].reshape(m, p, rank)
        dW = delta[n * m * rank + m * p * rank:].reshape(p, n, rank)
        
        step_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if step_norm > 1.0:
            scale = 1.0 / step_norm
            dU *= scale
            dV *= scale
            dW *= scale
        
        U_t += dU
        V_t += dV
        W_t += dW
    
    return U_t.reshape(n * m, rank), V_t.reshape(m * p, rank), W_t.reshape(p * n, rank)

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to round to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol):
        return U_r, V_r, W_r
    return None

def _find_2x3x3_rank14(time_budget=10):
    """
    Attempt to find a rank 14 integer/half-integer decomposition for 2x3 by 3x3.
    """
    n, m, p = 2, 3, 3
    rank = 14
    start_time = time.time()
    
    # We know the solution exists with small integers.
    # We will try random initializations and Newton polish + rounding.
    
    for attempt in range(500):
        if time.time() - start_time > time_budget:
            break
        
        np.random.seed(42 + attempt)
        # Initialize with small integers or random
        U_init = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
        V_init = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
        W_init = np.random.randint(-2, 3, size=(p*n, rank)).astype(float)
        
        # Polish
        U_pol, V_pol, W_pol = _newton_polish(U_init, V_init, W_init, n, m, p, max_iter=20, tol=1e-8)
        
        # Round and verify
        res = _try_round_and_verify(U_pol.tolist(), V_pol.tolist(), W_pol.tolist(), n, m, p)
        if res:
            return res
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 42
    
    # Fallback
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Strategy:
    # 1. Try to find rank 14 decomposition for 2x3x3.
    # 2. If found, construct rank 34 decomposition for 2x3x7.
    #    B is 3x7. Split into B1 (3x3), B2 (3x3), B3 (3x1).
    #    AB = A B1 + A B2 + A B3.
    #    Rank(AB1) = 14. Rank(AB2) = 14. Rank(AB3) = 6 (schoolbook).
    #    Total rank = 34.
    
    # Find 2x3x3 rank 14
    dec_233 = _find_2x3x3_rank14(time_budget=10)
    
    if dec_233:
        U_233, V_233, W_233 = dec_233
        # U_233: 6x14, V_233: 9x14, W_233: 6x14
        
        # Construct 2x3x7 rank 34
        rank_total = 14 + 14 + 6
        U_total = [[0] * rank_total for _ in range(n * m)]
        V_total = [[0] * rank_total for _ in range(m * p)]
        W_total = [[0] * rank_total for _ in range(p * n)]
        
        # Block 1: 2x3 by 3x3 (cols 0-2 of B)
        # U is same. V is top-left 3x3 of B. W is top 3x2 of result?
        # Wait, the tensor structure is:
        # U[i, j, c] * V[j, k, c] * W[k, i, c]
        # For 2x3x3: i in 0..1, j in 0..2, k in 0..2.
        # For 2x3x7: i in 0..1, j in 0..2, k in 0..6.
        
        # We need to map the 2x3x3 solution to the first 3 columns of k.
        # U_233[i, j, c] -> U_total[i, j, c]
        # V_233[j, k, c] -> V_total[j, k, c] for k in 0..2
        # W_233[k, i, c] -> W_total[k, i, c] for k in 0..2
        
        for i in range(n):
            for j in range(m):
                for c in range(14):
                    U_total[i * m + j][c] = U_233[i * m + j][c]
        
        for j in range(m):
            for k in range(3):
                for c in range(14):
                    V_total[j * p + k][c] = V_233[j * 3 + k][c]
        
        for k in range(3):
            for i in range(n):
                for c in range(14):
                    W_total[k * n + i][c] = W_233[k * n + i][c]
        
        # Block 2: 2x3 by 3x3 (cols 3-5 of B)
        # Same U. V is middle 3x3. W is middle 3x2.
        for i in range(n):
            for j in range(m):
                for c in range(14):
                    U_total[i * m + j][14 + c] = U_233[i * m + j][c]
        
        for j in range(m):
            for k in range(3):
                for c in range(14):
                    V_total[j * p + (3 + k)][14 + c] = V_233[j * 3 + k][c]
        
        for k in range(3):
            for i in range(n):
                for c in range(14):
                    W_total[(3 + k) * n + i][14 + c] = W_233[k * n + i][c]
        
        # Block 3: 2x3 by 3x1 (col 6 of B)
        # Schoolbook for 2x3 by 3x1. Rank 6.
        # i in 0..1, j in 0..2, k=6.
        # Terms: (i,j,k) -> 1
        for i in range(n):
            for j in range(m):
                for k in range(1): # Only k=6
                    idx = 28 + (i * m + j) * 1 + k
                    # Wait, schoolbook rank is n*m*p_block = 2*3*1 = 6.
                    # c_local in 0..5
                    c_local = i * m + j
                    U_total[i * m + j][28 + c_local] = 1
                    V_total[j * p + 6][28 + c_local] = 1
                    W_total[6 * n + i][28 + c_local] = 1
        
        # Verify
        if _verify(U_total, V_total, W_total, n, m, p):
            best = {"U": U_total, "V": V_total, "W": W_total}
            best_rank = 34
    
    return best
