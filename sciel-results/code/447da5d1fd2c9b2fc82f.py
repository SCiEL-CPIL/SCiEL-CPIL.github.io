import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=200, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _construct_strassen_56():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [0, 0, 0, 1], [1, 0, -1, 0], [0, -1, 0, 1],
        [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0], [1, -1, 1, 0, 0, 1, 0]
    ]
    
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    blocks = []
    for i in range(4):
        b_top = [0*8+2*i, 0*8+2*i+1, 1*8+2*i, 1*8+2*i+1]
        b_bot = [2*8+2*i, 2*8+2*i+1, 3*8+2*i, 3*8+2*i+1]
        c_idx = [2*i*2+0, 2*i*2+1, (2*i+1)*2+0, (2*i+1)*2+1]
        
        blocks.append((A1_idx, b_top, c_idx))
        blocks.append((A2_idx, b_bot, c_idx))
        
    col_offset = 0
    for a_idx, b_idx, c_idx in blocks:
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[a_idx[r]][col] = U_strassen[f][r]
                V[b_idx[r]][col] = V_strassen[f][r]
                W[c_idx[r]][col] = W_strassen[r][f]
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}

def _try_find_rank_48():
    n, m, p = 2, 4, 8
    rank = 48
    
    # Parameters: U (2*4*48), V (4*8*48), W (8*2*48)
    # Total params: 8*48 + 32*48 + 16*48 = 48 * (8+32+16) = 48*56 = 2688
    
    def residuals(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U_mat = U_flat.reshape(n*m, rank)
        V_mat = V_flat.reshape(m*p, rank)
        W_mat = W_flat.reshape(p*n, rank)
        
        # We need sum_c U[i,j,c] V[l,k,c] W[k,i,c] = delta_{j,l}
        # Residuals are the errors for all i, j, l, k
        res = []
        for i in range(n):
            for j in range(m):
                for l in range(m):
                    for k in range(p):
                        target = 1.0 if j == l else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U_mat[i*m+j, c] * V_mat[l*p+k, c] * W_mat[k*n+i, c]
                        res.append(val - target)
        return res

    # Initial guess: Random
    x0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank)
    
    # Use least_squares to minimize the residuals
    # This is a non-linear system.
    # We might need multiple restarts.
    
    start_time = time.time()
    deadline = start_time + 100
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        x0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank)
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=2000)
            if sol.cost < 1e-10:
                U_mat = sol.x[:n*m*rank].reshape(n*m, rank)
                V_mat = sol.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_mat = sol.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Snap to half-integers
                U_snap = np.round(U_mat * 2) / 2
                V_snap = np.round(V_mat * 2) / 2
                W_snap = np.round(W_mat * 2) / 2
                
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                U_final = [[float(x) for x in row] for row in U_list]
                V_final = [[float(x) for x in row] for row in V_list]
                W_final = [[float(x) for x in row] for row in W_list]
                
                if _verify(U_final, V_final, n=2, m=4, p=8, trials=50):
                    return {"U": U_final, "V": V_final, "W": W_final}
        except Exception:
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a rank-48 solution
    sol_48 = _try_find_rank_48()
    if sol_48 is not None:
        if _verify(sol_48["U"], sol_48["V"], sol_48["W"], trials=10):
            return sol_48
            
    # Fallback to 56
    sol = _construct_strassen_56()
    if _verify(sol["U"], sol["V"], sol["W"], trials=10):
        return sol
    else:
        n, m, p = 2, 4, 8
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
