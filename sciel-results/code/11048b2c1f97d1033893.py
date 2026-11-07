import numpy as np
import time
from scipy.optimize import differential_evolution

N = 50
# The domain for h is indices 0..49 corresponding to [-1/4, 1/4]
# x_i = -0.25 + i * (0.5 / 49)
x_grid = np.linspace(-0.25, 0.25, N)

def _eval(h):
    """
    Computes the objective value for a given height vector h.
    h is assumed to be non-negative.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    if np.sum(h) == 0:
        return 0.0
        
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def _gmm_to_h(params):
    """
    Convert Gaussian Mixture Model parameters to heights h.
    params: [A1, c1, s1, A2, c2, s2, A3, c3, s3]
    K=3 Gaussians.
    A: amplitude, c: center, s: sigma.
    """
    K = 3
    h = np.zeros(N)
    for i in range(K):
        A = params[i*3]
        c = params[i*3 + 1]
        s = params[i*3 + 2]
        # Ensure sigma is positive
        s = max(abs(s), 0.01)
        h += A * np.exp(-0.5 * ((x_grid - c) / s) ** 2)
    
    # Enforce non-negativity (though Gaussians with A>0 are positive, 
    # we might allow negative A to create "dips" if we wanted, but task says non-negative heights.
    # So we force A > 0. But GMM with positive A is always positive.
    # If we allow A to be negative, we must clip.
    # Let's allow A to be any real, and clip the result to [0, inf).
    # This allows for more complex shapes like "donut" if we had multiple, 
    # but with 3 positive bumps, we get multi-modal peaks.
    
    h = np.clip(h, 0, None)
    
    # Normalize to sum=1 for scale invariance stability
    s_sum = np.sum(h)
    if s_sum == 0:
        return np.ones(N) / N
    h = h / s_sum
    
    return h

def _objective(params):
    """
    Objective function for minimization (returns negative value).
    """
    try:
        h = _gmm_to_h(params)
        val = _eval(h)
        return -val
    except Exception:
        return 0.0 # Large penalty

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # Safety margin for 150s limit
    
    # Define bounds for the 9 parameters: [A1, c1, s1, A2, c2, s2, A3, c3, s3]
    # Amplitudes: [0, 10] (relative, will be normalized)
    # Centers: [-0.25, 0.25]
    # Sigmas: [0.01, 0.5]
    
    bounds = [
        (0.0, 10.0), (-0.25, 0.25), (0.01, 0.5),
        (0.0, 10.0), (-0.25, 0.25), (0.01, 0.5),
        (0.0, 10.0), (-0.25, 0.25), (0.01, 0.5)
    ]
    
    best_params = None
    best_val = -np.inf
    
    # Run Differential Evolution
    # Use 'best1bin' strategy, moderate popsize for speed
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            polish=False # We will polish separately to save time or if DE is slow
        )
        
        if res.fun < best_val:
            best_val = -res.fun
            best_params = res.x.copy()
            
    except Exception:
        pass
        
    # If time allows, try a few more restarts with different seeds or strategies
    if time.time() < deadline - 20:
        for seed in [123, 456]:
            if time.time() > deadline - 10:
                break
            try:
                res2 = differential_evolution(
                    _objective, 
                    bounds, 
                    maxiter=100, 
                    popsize=15, 
                    tol=1e-8, 
                    seed=seed,
                    polish=False
                )
                if res2.fun < best_val:
                    best_val = -res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # If we have significant time left, polish the best solution using L-BFGS-B
    if best_params is not None and time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            res_polish = minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res_polish.fun < best_val:
                best_val = -res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # If DE failed completely, fallback to a simple heuristic
    if best_params is None:
        # Try a single central Gaussian
        best_params = np.array([1.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1])
        best_val = -_objective(best_params)

    # Convert best params to heights
    best_h = _gmm_to_h(best_params)
    
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
