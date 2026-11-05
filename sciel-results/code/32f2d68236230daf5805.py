import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    def generate_h(params):
        phi0, mu = params
        k = np.arange(N)
        # Use a wider range for mu to allow faster chirps
        h = np.cos(phi0 + mu * k**2)
        return h

    def objective(params):
        h = generate_h(params)
        sum_h = np.sum(h)
        if abs(sum_h) < 1e-8:
            return 1e10
        conv_h = np.convolve(h, h)
        max_conv = np.max(conv_h)
        value = abs(2 * N * max_conv / (sum_h ** 2))
        return value

    # 1. Grid Search on log-mu and phi0
    # This ensures we cover the parameter space robustly
    log_mu_range = np.linspace(np.log(1e-4), np.log(1e3), 50)
    phi0_range = np.linspace(0, 2 * np.pi, 20)
    
    best_val = np.inf
    best_params = None
    
    for log_mu in log_mu_range:
        mu = np.exp(log_mu)
        for phi0 in phi0_range:
            val = objective([phi0, mu])
            if val < best_val:
                best_val = val
                best_params = [phi0, mu]
    
    # 2. Local Refinement using Nelder-Mead
    # Start from the best grid search result
    if best_params is not None:
        result_nm = minimize(
            objective, 
            best_params, 
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000}
        )
        
        if result_nm.fun < best_val:
            best_val = result_nm.fun
            best_params = result_nm.x

    # 3. Try a few other random restarts for NM to catch other local minima
    rng = np.random.default_rng(42)
    for _ in range(10):
        p0 = [rng.uniform(0, 2*np.pi), np.exp(rng.uniform(np.log(1e-4), np.log(1e3)))]
        res = minimize(
            objective, 
            p0, 
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000, 'maxfev': 5000}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # Final check and construction
    h = generate_h(best_params)
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-6:
        return np.ones(N)
        
    conv_h = np.convolve(h, h)
    max_conv = np.max(conv_h)
    val = abs(2 * N * max_conv / (sum_h ** 2))
    
    # Compare with seed (value 2.0)
    if val < 2.0:
        return h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
