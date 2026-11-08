import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Optimization Strategy: Multi-start Powell/Nelder-Mead
    # Powell is often better for small N and non-smooth (though this is smooth)
    # We use log_roots to ensure positivity
    
    log_low = -2.0
    log_high = 12.0
    bounds = [(log_low, log_high)] * 12
    
    best_val = 1e10
    best_log_roots = None
    best_roots = None
    
    # Initial guess: Seed
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    seed_log_roots = np.clip(seed_log_roots, log_low, log_high)
    
    # Evaluate seed
    val = cost_func(seed_log_roots)
    if val < best_val:
        best_val = val
        best_log_roots = seed_log_roots.copy()
        best_roots = np.exp(seed_log_roots)

    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # We will run several restarts.
    # Powell is deterministic given start, so random starts are key.
    # Nelder-Mead is also good.
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Generate random start
        # Mix of uniform and seed-perturbed
        if i % 2 == 0:
            init = rng.uniform(log_low, log_high, size=12)
        else:
            init = seed_log_roots + rng.normal(0, 1.0, size=12)
            init = np.clip(init, log_low, log_high)
            
        try:
            # Try Powell first, it's robust
            res = minimize(
                cost_func, 
                init, 
                method='Powell',
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(res.x)
                
            # Polish with Nelder-Mead or L-BFGS-B
            if time.time() < deadline - 10:
                res2 = minimize(
                    cost_func, 
                    res.x, 
                    method='L-BFGS-B',
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_log_roots = res2.x
                    best_roots = np.exp(res2.x)
                    
        except Exception:
            pass

    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    return [4.0 * (2.0 ** i) for i in range(12)]
