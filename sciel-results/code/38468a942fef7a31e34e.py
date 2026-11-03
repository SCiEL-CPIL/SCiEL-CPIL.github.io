import numpy as np
from scipy.optimize import minimize, differential_evolution
from math import factorial
import time

def _compute_bound_L2(c0, c4, c8):
    """
    Compute the L2 uncertainty product A(g)*A(hat_g) for g(x) = P(x)exp(-x^2/2).
    """
    # c12 is determined by g(0)=0
    c12 = -(c0 * 1.0 + c4 * 3.0 + c8 * 105.0) / 10395.0
    
    coeffs = {0: c0, 4: c4, 8: c8, 12: c12}
    
    # Calculate norm squared: sum c_k^2 * 2^k * k!
    norm_sq = 0.0
    for k, c in coeffs.items():
        if c != 0:
            norm_sq += (c * c) * (2.0 ** k) * (float(factorial(k)))
            
    if norm_sq < 1e-15:
        return float('inf')
        
    # Calculate second moment: int x^2 P(x)^2 e^{-x^2} dx
    # x H_k = 0.5 H_{k+1} + k H_{k-1}
    d = {}
    for k, c in coeffs.items():
        if c == 0:
            continue
        k1 = k + 1
        d[k1] = d.get(k1, 0.0) + 0.5 * c
        k_minus_1 = k - 1
        if k_minus_1 >= 0:
            d[k_minus_1] = d.get(k_minus_1, 0.0) + k * c
            
    second_moment = 0.0
    for j, dj in d.items():
        if dj != 0:
            second_moment += (dj * dj) * (2.0 ** j) * (float(factorial(j)))
            
    return second_moment / norm_sq

def construct():
    deadline = time.time() + 170.0
    
    def objective(x):
        return _compute_bound_L2(x[0], x[1], x[2])
        
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_x = None
    best_val = float('inf')
    
    # Differential Evolution for global search
    try:
        if time.time() < deadline - 10:
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=200, 
                popsize=20, 
                tol=1e-10, 
                seed=42,
                polish=False
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Local Polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                objective, 
                best_x, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Fallback: Random restarts if DE failed
    if best_x is None or best_val > 1.0:
        np.random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            x0 = np.random.uniform(-5, 5, 3)
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass
                
    if best_x is None:
        return np.array([0.0, 0.0, 0.0])
        
    return np.array(best_x)
