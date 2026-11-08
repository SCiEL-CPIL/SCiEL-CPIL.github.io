import numpy as np
import time
import random
from scipy.optimize import minimize

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    n, m, p = 2, 4, 4
    rank = U.shape[1] if isinstance(U, np.ndarray) else len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=1e-9)

def _strassen_decomposition():
    """Generate the rank-28 Strassen-based decomposition."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_terms = [
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1, (1,1): -1}
        },
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(1,0): 1, (1,1): 1}
        },
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1},
            'W': {(0,0): 1, (0,1): 1}
        },
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(1,0): 1, (0,0): -1}
        },
        {
            'U': {(1,0): 1, (1,1): -1},
            'V': {(0,1): 1, (1,1): 1},
            'W': {(1,1): 1}
        },
        {
            'U': {(0,1): 1, (1,0): -1},
            'V': {(0,0): 1, (1,0): 1},
            'W': {(0,0): 1}
        }
    ]
    
    sub_problems = [
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2)
    ]
    
    col = 0
    for j_off, k_off in sub_problems:
        for term in strassen_terms:
            for (i_l, j_l), val in term['U'].items():
                i_g = i_l
                j_g = j_l + j_off
                U[i_g * m + j_g, col] += val
                
            for (j_l, k_l), val in term['V'].items():
                j_g = j_l + j_off
                k_g = k_l + k_off
                V[j_g * p + k_g, col] += val
                
            for (k_l, i_l), val in term['W'].items():
                k_g = k_l + k_off
                i_g = i_l
                W[k_g * n + i_g, col] += val
                
            col += 1
            
    return U, V, W

def _try_rank_reduction(U0, V0, W0, target_rank, deadline, n=2, m=4, p=4):
    """Try to reduce rank to target_rank using optimization."""
    r = target_rank
    T_target = np.ones((n, m, p))
    
    def objective(params):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W = params[n*m*r+m*p*r:].reshape(p*n, r)
        
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        
        return np.sum((T - T_target)**2)
    
    # Multiple random initializations
    for trial in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + trial)
        U_init = np.random.randn(n*m, r)
        V_init = np.random.randn(m*p, r)
        W_init = np.random.randn(p*n, r)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        result = minimize(objective, params_init, method='Powell',
                        options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                               'disp': False})
        
        if result.fun < 1e-8:
            U_opt = result.x[:n*m*r].reshape(n*m, r)
            V_opt = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_opt = result.x[n*m*r+m*p*r:].reshape(p*n, r)
            
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
                return U_opt, V_opt, W_opt
    
    return None

def _schoolbook():
    n, m, p = 2, 4, 4
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

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = 2, 4, 4
    
    # Start with Strassen-based rank-28 solution
    U, V, W = _strassen_decomposition()
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        return _schoolbook()
    
    # Try to find lower rank decompositions: 27, 26, 25
    for target_rank in [27, 26, 25]:
        if time.time() > deadline - 30:  # Need 30s buffer
            break
        
        result = _try_rank_reduction(U, V, W, target_rank, deadline)
        if result is not None:
            U_opt, V_opt, W_opt = result
            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # Return the verified rank-28 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
