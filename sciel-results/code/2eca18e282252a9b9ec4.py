import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 140.0

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

def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Fraction(V[i][c]) for c in range(rank)] for i in range(m * p)]
    Wf = [[Fraction(W[i][c]) for c in range(rank)] for i in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if s != Fraction(1):
                    return False
    return True

def _verify(n, m, p, U, V, W):
    """Fast float-based verification for initial checks."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses structured initialization and high-precision numerical optimization,
    followed by exact rational verification.
    """
    if time.time() > _DEADLINE - 25:
        return None

    target_rank = 49
    try:
        import scipy.optimize as opt
        
        best_error = float('inf')
        best_params = None
        
        # Structured initialization: start from schoolbook and perturb
        max_restarts = 8
        
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 8:
                break
            
            if restart == 0:
                # Schoolbook-based initialization: use 49 of the 50 schoolbook terms
                # and try to adjust them
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                c = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if c < target_rank:
                                U[i * m + j][c] = 1.0
                                V[j * p + k][c] = 1.0
                                W[k * n + i][c] = 1.0
                            c += 1
                # Add small perturbations to allow the optimizer to find a solution
                np.random.seed(42)
                U += np.random.randn(n * m, target_rank) * 0.1
                V += np.random.randn(m * p, target_rank) * 0.1
                W += np.random.randn(p * n, target_rank) * 0.1
            else:
                # Random initialization with structure
                np.random.seed(100 + restart * 7)
                U = np.random.randn(n * m, target_rank) * 0.3
                V = np.random.randn(m * p, target_rank) * 0.3
                W = np.random.randn(p * n, target_rank) * 0.3
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                U = U_flat.reshape(n*m, target_rank)
                V = V_flat.reshape(m*p, target_rank)
                W = W_flat.reshape(p*n, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            error += (s - 1.0)**2
                return error
            
            try:
                res = opt.minimize(objective, params, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x.copy()
                    # Print progress for debugging
                    if best_error < 1e-6:
                        print(f"Restart {restart}: error = {best_error:.2e}")
                    
            except Exception:
                pass
        
        if best_params is not None and best_error < 1e-6:
            # High-precision polish using the best solution
            U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try to find exact rational solution by searching nearby rational points
            # Use larger denominators for better chance of exact match
            for denom in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                if time.time() > _DEADLINE - 5:
                    break
                
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                # Quick float check first
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    # Exact verification
                    if _verify_exact(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Try local perturbations around the rounded values
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            if time.time() > _DEADLINE - 3:
                                break
                            U_p = U_r + di / denom
                            V_p = V_r + dj / denom
                            W_p = W_r + dk / denom
                            if _verify(n, m, p, U_p.tolist(), V_p.tolist(), W_p.tolist()):
                                if _verify_exact(n, m, p, U_p.tolist(), V_p.tolist(), W_p.tolist()):
                                    return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
        
    except Exception:
        pass
        
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify_exact(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
