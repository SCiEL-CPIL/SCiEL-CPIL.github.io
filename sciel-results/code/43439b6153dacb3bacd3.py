import numpy as np
import time
from fractions import Fraction
from scipy.optimize import least_squares

def find_decomp_223():
    """
    Finds a rank 10 decomposition for 2x2x3 matrix multiplication.
    Returns U, V, W as lists of lists of Fractions or None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    def residuals(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_sol = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 10
    
    for seed in range(100):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        init_size = n*m*rank + m*p*rank + p*n*rank
        x0 = np.random.uniform(-1, 1, init_size)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=2000)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_sol = sol.x
                if err < 1e-12:
                    break
        except Exception:
            continue
            
    if best_err > 1e-9 or best_sol is None:
        return None

    U_flat = best_sol[:n*m*rank]
    V_flat = best_sol[n*m*rank:n*m*rank + m*p*rank]
    W_flat = best_sol[n*m*rank + m*p*rank:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    U_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if val != 1:
                    return None
                    
    return U_frac, V_frac, W_frac

def find_decomp_243_rank19():
    """
    Finds a rank 19 decomposition for 2x4x3 matrix multiplication.
    Returns U, V, W as lists of lists of Fractions or None.
    """
    n, m, p = 2, 4, 3
    rank = 19
    total = n * m * p  # 24 constraints
    
    def residuals(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        res = np.zeros(total)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    best_sol = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 30
    
    # Initialize with random values
    for seed in range(200):
        if time.time() > deadline:
            break
        if best_err < 1e-12:
            break
            
        np.random.seed(seed)
        init_size = n*m*rank + m*p*rank + p*n*rank
        x0 = np.random.uniform(-1, 1, init_size)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=3000)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_sol = sol.x
                if err < 1e-12:
                    break
        except Exception:
            continue
            
    if best_sol is None or best_err > 1e-9:
        return None

    U_flat = best_sol[:n*m*rank]
    V_flat = best_sol[n*m*rank:n*m*rank + m*p*rank]
    W_flat = best_sol[n*m*rank + m*p*rank:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    U_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if val != 1:
                    return None
                    
    return U_frac, V_frac, W_frac

def find_decomp_246_rank40():
    # ... (keep existing) ...
    pass # Placeholder, will be replaced in edit

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    global_deadline = start_time + 140
    
    # Try rank-38 via 2x4x3 rank-19
    decomp_243 = find_decomp_243_rank19()
    
    if decomp_243 is not None:
        U243, V243, W243 = decomp_243
        rank = 38
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # First block: k=0,1,2
        # U indices: i*4+j (0..7)
        # V indices: j*3+k (0..11)
        # W indices: k*2+i (0..5)
        
        for i in range(2):
            for j in range(4):
                for k in range(3):
                    for r in range(19):
                        u_idx = i*4 + j
                        v_idx = j*3 + k
                        w_idx = k*2 + i
                        
                        U[u_idx][r] += U243[i*4+j][r]
                        V[v_idx][r] += V243[j*3+k][r]
                        W[w_idx][r] += W243[k*2+i][r]
                        
        # Second block: k=3,4,5
        # W indices for second block: (k-3)*2 + i
        for i in range(2):
            for j in range(4):
                for k_local in range(3):
                    k = k_local + 3
                    for r in range(19):
                        u_idx = i*4 + j
                        v_idx = j*3 + k
                        w_idx = k_local*2 + i
                        
                        U[u_idx][19 + r] += U243[i*4+j][r]
                        V[v_idx][19 + r] += V243[j*3+k_local][r]
                        W[w_idx][19 + r] += W243[k_local*2+i][r]
                        
        return {"U": U, "V": V, "W": W}
    
    # Fallback to previous logic
    # ...
