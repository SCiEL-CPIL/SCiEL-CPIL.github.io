import time
import numpy as np
from scipy.optimize import differential_evolution

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1) > tol:
                    return False
    return True

def _round_to_half(x):
    return np.round(2 * x) / 2.0

def _objective(params, n=3, m=3, p=3, r=23):
    """
    Objective function for differential evolution.
    Minimizes the squared error of the tensor decomposition.
    Also adds a penalty for values that are far from half-integers.
    """
    U = params[0:n*m*r].reshape(n*m, r)
    V = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W = params[n*m*r+m*p*r:].reshape(p*n, r)
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                err += (val - 1.0) ** 2
    
    # Penalty for non-half-integer values
    # We want U, V, W to be close to k/2
    # dist = |2*x - round(2*x)|
    penalty = 0.0
    for arr in [U, V, W]:
        penalty += np.sum((2 * arr - np.round(2 * arr)) ** 2)
        
    return err + 100 * penalty

def _find_decomposition_de(r, n=3, m=3, p=3, time_budget=60.0):
    """
    Try to find a rank-r decomposition using Differential Evolution.
    """
    start = time.time()
    
    # Define bounds for DE. 
    # Values are expected to be small, roughly in [-2, 2]
    bounds = [(-2.0, 2.0)] * (n*m*r + m*p*r + p*n*r)
    
    # DE is stochastic, so we run it with a specific seed for reproducibility if needed, 
    # but we want to explore.
    
    try:
        # maxiter and popsize are critical for time budget.
        # 23 * 27 * 3 = 1863 variables.
        # DE is slow for high dimensions.
        # Maybe r=23 is too high for DE to converge in time?
        # Let's try r=24 or 25 first? No, we want minimal.
        
        # If r is too high, DE might not converge.
        # We will try r=23.
        
        res = differential_evolution(
            _objective, 
            bounds, 
            args=(n, m, p, r),
            maxiter=100, 
            popsize=10, 
            tol=1e-8,
            seed=42,
            polish=True
        )
        
        if time.time() - start > time_budget:
            return None

        if res.fun < 1e-6:
            best_params = res.x
            U = best_params[0:n*m*r].reshape(n*m, r)
            V = best_params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W = best_params[n*m*r+m*p*r:].reshape(p*n, r)
            
            U_rounded = _round_to_half(U)
            V_rounded = _round_to_half(V)
            W_rounded = _round_to_half(W)
            
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
                
            # Try integer rounding
            U_int = np.round(U).astype(np.float64)
            V_int = np.round(V).astype(np.float64)
            W_int = np.round(W).astype(np.float64)
            
            if _verify(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
                
    except Exception as e:
        pass
        
    return None

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Guaranteed rank-27 decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find rank-23 decomposition
    # DE is slow, so we give it a tight budget.
    if time.time() - start_time < timeout - 5:
        time_left = timeout - (time.time() - start_time) - 5
        # If time is tight, DE might not finish. 
        # Let's try a smaller rank first? No, 23 is the target.
        # Maybe try r=24 which is easier?
        # Actually, let's try r=23.
        
        # To make it faster, we can reduce the population size or iterations.
        # But DE is stochastic.
        
        # Let's try a quick local search from a good start instead of DE?
        # No, DE is the "stronger algorithm" requested.
        
        result = _find_decomposition_de(23, n, m, p, time_budget=time_left)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = 23
            return best_result
            
    # Try rank 24
    if time.time() - start_time < timeout - 2:
        time_left = timeout - (time.time() - start_time) - 2
        result = _find_decomposition_de(24, n, m, p, time_budget=time_left)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = 24
            return best_result
            
    return best_result
