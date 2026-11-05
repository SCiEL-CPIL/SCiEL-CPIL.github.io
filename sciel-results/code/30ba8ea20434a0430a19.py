import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    def generate_h(params):
        phi0, mu = params
        k = np.arange(N)
        # Linearly varying frequency implies quadratic phase
        # h = cos(phi0 + mu * k**2)
        # We use mu directly as the coefficient of k^2
        h = np.cos(phi0 + mu * k**2)
        return h

    def objective(params):
        h = generate_h(params)
        sum_h = np.sum(h)
        if abs(sum_h) < 1e-10:
            return 1e10
        conv_h = np.convolve(h, h)
        # np.max takes the maximum signed entry
        max_conv = np.max(conv_h)
        value = abs(2 * N * max_conv / (sum_h ** 2))
        return value

    # Define bounds
    # phi0 in [0, 2*pi)
    # mu in [1e-5, 10]
    # Log scale for mu might be better, but DE handles linear bounds.
    # We can transform mu to log_mu in DE, but let's try linear first with wider range.
    bounds = [(0, 2 * np.pi), (1e-5, 10.0)]
    
    # Global Search
    # Use a seed for reproducibility
    result_de = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=300, 
        tol=1e-8, 
        popsize=40, 
        init='latinhypercube',
        polish=False 
    )
    
    best_val = result_de.fun
    best_params = result_de.x
    
    # Local Refinement
    # Use Nelder-Mead for refinement
    result_nm = minimize(
        objective, 
        best_params, 
        method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000}
    )
    
    if result_nm.fun < best_val:
        best_val = result_nm.fun
        best_params = result_nm.x

    # Additional restarts to escape local minima
    rng = np.random.default_rng(123)
    for _ in range(20):
        p0 = [rng.uniform(0, 2*np.pi), rng.uniform(1e-5, 10.0)]
        res = minimize(
            objective, 
            p0, 
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 2000}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # Final check
    h = generate_h(best_params)
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-6:
        return np.ones(N)
        
    conv_h = np.convolve(h, h)
    max_conv = np.max(conv_h)
    val = abs(2 * N * max_conv / (sum_h ** 2))
    
    if val < 2.0:
        return h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
