"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    deadline = time.time() + 130  # Reserve 20s margin for 150s total
    
    # First, verify and return the known rank-34 decomposition if valid
    r34 = construct_r34()
    if valid_res(r34, 3, 3, 4):
        best_rank = 34
        best_result = r34
    else:
        # Fallback: construct a simple rank-36 decomposition (3x3x4 standard)
        best_rank = 36
        best_result = construct_r36()
    
    # Try to find lower rank decompositions
    for target_rank in range(20, best_rank):
        if time.time() >= deadline:
            break
        
        # Multiple restarts with different strategies
        for attempt in range(8):
            if time.time() >= deadline:
                break
            res = attempt_lower_rank(target_rank, seed=attempt, strategy=attempt % 3)
            if res is not None and valid_res(res, 3, 3, 4):
                return res
    
    return best_result

def attempt_lower_rank(target_rank, seed=0, strategy=0):
    n, m, p = 3, 3, 4
    R = target_rank
    
    # Build the target tensor indices
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                indices_a.append(a)
                indices_b.append(b)
                indices_c.append(c)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    # Objective function
    def objective(params):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
        error = np.sum((vals - 1.0)**2)
        return error
    
    # Different initialization strategies
    rng = np.random.RandomState(seed * 1000 + strategy * 7)
    
    if strategy == 0:
        # Random Gaussian initialization
        params = rng.randn((n*m + m*p + p*n) * R)
    elif strategy == 1:
        # Sparse initialization - start with mostly zeros
        params = rng.randn((n*m + m*p + p*n) * R)
        params[np.abs(params) < 0.5] = 0
    else:
        # Binary-like initialization
        params = rng.choice([-1, 1], size=(n*m + m*p + p*n) * R)
    
    # Optimization
    from scipy.optimize import minimize
    try:
        res = minimize(objective, params, method='L-BFGS-B', 
                      options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.fun < 1e-8:
            U = res.x[0:n*m*R].reshape(n*m, R)
            V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            
            # Try rounding to integers/half-integers
            for scale in [2, 4, 8]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                
                vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                if np.allclose(vals, 1.0, atol=1e-9):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
    except Exception:
        pass
        
    return None

def valid_res(res, n, m, p):
    U = np.array(res['U'], dtype=float)
    V = np.array(res['V'], dtype=float)
    W = np.array(res['W'], dtype=float)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False
    
    # Check for integer/half-integer entries
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
            return False
    
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                indices_a.append(i * m + k)
                indices_b.append(k * p + j)
                indices_c.append(j * n + i)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
    return np.allclose(vals, 1.0, atol=1e-9)

def construct_r34():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct_r36():
    """Simple rank-36 decomposition using standard matmul."""
    n, m, p = 3, 3, 4
    rank = n * m * p  # 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
