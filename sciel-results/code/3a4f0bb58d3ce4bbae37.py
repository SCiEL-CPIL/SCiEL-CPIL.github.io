import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Try to find rank 15 decomposition with integer coefficients
    # Using a structured search
    if _try_rank_15_construct(n, m, p, deadline):
        pass  # Should have returned already
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def _try_rank_15_construct(n, m, p, deadline):
    """Try to construct a rank-15 decomposition with integer/half-integer entries."""
    # For 2x3 * 3x3, we know rank <= 15 is possible.
    # Let's try a specific construction.
    
    # One known approach: use the fact that we can factor the 3x3 multiplication
    # into smaller operations.
    
    # Actually, let me try a direct search with small integer values.
    # The search space is huge, so let's be smart about it.
    
    # For rank 15, we need 15 columns in U (6x15), V (9x15), W (6x15).
    # Each entry is in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2} or similar small set.
    
    # This is still too large for exhaustive search. Let's try a different approach:
    # Use the structure of the problem to reduce the search.
    
    # Alternative: try known algebraic constructions.
    # For 2x3 * 3x3, one can use a variant of Strassen's algorithm.
    
    # Let me try to hardcode a known rank-15 decomposition if I can find one,
    # or use a more efficient search.
    
    # Actually, let's try a randomized approach with better initialization and
    # more iterations, using scipy.optimize with multiple methods.
    
    from scipy.optimize import minimize, differential_evolution
    
    # Target tensor
    T = np.zeros((n, m, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
    
    # Try rank 15 with differential evolution (global optimizer)
    rank = 15
    nm, mp, pn = n*m, m*p, p*n
    total_params = (nm + mp + pn) * rank
    
    def objective(params):
        U_r = params[:nm*rank].reshape(n, m, rank)
        V_r = params[nm*rank: nm*rank+mp*rank].reshape(m, p, rank)
        W_r = params[nm*rank+mp*rank:].reshape(p, n, rank)
        
        T_pred = np.einsum('ijc,jkc,kpc->ijkkpc', U_r, V_r, W_r, optimize=True)
        diff = T_pred - T
        return np.sum(diff * diff)
    
    # Differential evolution with bounds
    bounds = [(-2, 2)] * total_params
    
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=200, popsize=15, seed=42,
            tol=1e-10, mutation=(0.3, 1.0), recombination=0.7,
            polish=True, init='latinhypercube'
        )
        
        if result.fun < 1e-8:
            U_sol = result.x[:nm*rank].reshape(nm, rank)
            V_sol = result.x[nm*rank: nm*rank+mp*rank].reshape(mp, rank)
            W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Try rounding to half-integers
            U_int = np.round(U_sol * 2) / 2
            V_int = np.round(V_sol * 2) / 2
            W_int = np.round(W_sol * 2) / 2
            
            if _verify(U_int, V_int, W_int, n, m, p):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
            # Try rounding to integers
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            if _verify(U_int, V_int, W_int, n, m, p):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
    except Exception:
        pass
    
    # If time allows, try more restarts
    if time.time() < deadline - 5:
        for seed in range(5):
            if time.time() > deadline - 2:
                break
            try:
                result = differential_evolution(
                    objective, bounds,
                    maxiter=100, popsize=10, seed=seed+100,
                    tol=1e-10, mutation=(0.5, 1.0), recombination=0.8,
                    polish=True, init='latinhypercube'
                )
                
                if result.fun < 1e-8:
                    U_sol = result.x[:nm*rank].reshape(nm, rank)
                    V_sol = result.x[nm*rank: nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    if _verify(U_int, V_int, W_int, n, m, p):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    if _verify(U_int, V_int, W_int, n, m, p):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                continue
    
    return None

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    if not np.allclose(U * 2, np.round(U * 2)): return False
    if not np.allclose(V * 2, np.round(V * 2)): return False
    if not np.allclose(W * 2, np.round(W * 2)): return False
    
    U_reshaped = np.array(U).reshape(n, m, r)
    V_reshaped = np.array(V).reshape(m, p, r)
    W_reshaped = np.array(W).reshape(p, n, r)
    
    T_pred = np.einsum('ijc,jkc,kpc->ijkkpc', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    T = np.zeros((n, m, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
    
    return np.allclose(T_pred, T, atol=1e-9)
