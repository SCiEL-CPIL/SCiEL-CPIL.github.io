import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def _verify_solution(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                    target = 1.0 if j == jp else 0.0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _polish(U, V, W, n, m, p, rank, max_nfev=1000):
    """
    Polishes the solution using scipy.optimize.least_squares.
    """
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def residual(x):
        U_res = x[:n*m*rank].reshape(n*m, rank)
        V_res = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_res = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = np.dot(W_res[k*n+i], U_res[i*m+j] * V_res[jp*p+k])
                        target = 1.0 if j == jp else 0.0
                        res.append(s - target)
        return np.array(res)
        
    try:
        res = least_squares(residual, x0, method='trf', max_nfev=max_nfev)
        x_opt = res.x
        U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
        V_opt = x_opt[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = x_opt[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        if np.linalg.norm(res.fun) < 1e-9:
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
    except Exception:
        pass
    return None

def _als_solve(rank, n, m, p, max_iters=200, restarts=50, timeout=30):
    start_time = time.time()
    
    # Precompute index arrays for vectorization
    # For U update: for each (i,j), we need A[k*m+jp] = W[k*n+i] * V[jp*p+k]
    # For V update: for each (j,k), we need A[i*m2+jp] = W[k*n+i] * U[i*m+j']
    # For W update: for each (k,i), we need A[j*m+jp] = U[i*m+j] * V[jp*p+k]
    
    for restart in range(restarts):
        if time.time() - start_time > timeout:
            break
            
        # Use different initialization strategies
        strategy = restart % 4
        if strategy == 0:
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
        elif strategy == 1:
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
        elif strategy == 2:
            # Sparse initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for c in range(rank):
                U[np.random.randint(0, n*m), c] = np.random.choice([-1, 1])
                V[np.random.randint(0, m*p), c] = np.random.choice([-1, 1])
                W[np.random.randint(0, p*n), c] = np.random.choice([-1, 1])
        else:
            # Mixed scale
            U = np.random.randn(n * m, rank) * 0.7
            V = np.random.randn(m * p, rank) * 0.7
            W = np.random.randn(p * n, rank) * 0.7
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
                
            # Update U - vectorized
            for i in range(n):
                for j in range(m):
                    # Build A matrix: rows indexed by (k, jp), A[k*m+jp] = W[k*n+i] * V[jp*p+k]
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = x
                    
            # Update V - vectorized
            for j in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * U[i * m + jp])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = x

            # Update W - vectorized
            for k in range(p):
                for i in range(n):
                    A_rows = []
                    b_vals = []
                    for j in range(m):
                        for jp in range(m):
                            A_rows.append(U[i * m + j] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = x

            # Check error - vectorized
            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = np.dot(W[k * n + i], U[i * m + j] * V[jp * p + k])
                            target = 1.0 if j == jp else 0.0
                            error += (s - target) ** 2
            
            if error < 1e-10:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_solution(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
            
            if error < 1e-4 and it > 10:
                polished = _polish(U, V, W, n, m, p, rank, max_nfev=200)
                if polished:
                    U_list, V_list, W_list = polished
                    if _verify_solution(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list

    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 140
    
    # Known lower bounds suggest rank >= 19 for 2x3*3x4
    # Try ranks in order of priority
    ranks_to_try = [19, 20, 21, 22, 23]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Allocate time: more time for lower ranks (harder to find)
        if r <= 20:
            t_limit = min(45, remaining - 2)
        elif r <= 22:
            t_limit = min(20, remaining - 2)
        else:
            t_limit = min(10, remaining - 2)
            
        if t_limit < 1:
            continue

        result = _als_solve(r, n, m, p, max_iters=500, restarts=200, timeout=t_limit)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    return _schoolbook()
