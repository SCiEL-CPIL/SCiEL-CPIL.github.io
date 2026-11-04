import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses numerical optimization to find a real-valued solution, then rounds to rationals.
    """
    if time.time() > _DEADLINE - 20:
        return None

    target_rank = 49
    # Dimensions: n=2, m=5, p=5
    # U: (10, 49), V: (25, 49), W: (10, 49)
    
    try:
        import scipy.optimize as opt
        
        best_error = float('inf')
        best_params = None
        
        # Number of restarts depends on time
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 5:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                U = U_flat.reshape(n*m, target_rank)
                V = V_flat.reshape(m*p, target_rank)
                W = W_flat.reshape(p*n, target_rank)
                
                # Vectorized computation of the tensor contraction
                # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            error += (s - 1.0)**2
                return error
            
            try:
                # Use L-BFGS-B for better convergence on smooth problems
                res = opt.minimize(objective, params, method='L-BFGS-B', 
                                   options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
                    
            except Exception:
                pass
        
        if best_params is not None:
            # Round to rationals
            U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try multiple rounding granularities
            for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                # Verify exactly
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Also try with slight perturbations for near-misses
                for delta in [-0.5, 0.5]:
                    U_r2 = U_r + delta / denom
                    V_r2 = V_r + delta / denom
                    W_r2 = W_r + delta / denom
                    if _verify(n, m, p, U_r2.tolist(), V_r2.tolist(), W_r2.tolist()):
                        return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
        
    except Exception:
        pass
        
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
