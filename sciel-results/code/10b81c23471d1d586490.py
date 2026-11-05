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

    # 1. Global Search using Differential Evolution
    # We search over phi0 in [0, 2pi) and mu in a log-scale range
    # mu determines the "chirp rate".
    
    # Use a robust DE configuration
    bounds = [(0, 2 * np.pi), (1e-4, 1e3)]
    
    # Run DE with a large population and many iterations to ensure convergence
    # Since the dimension is only 2, we can afford a larger population
    result_de = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=500, 
        tol=1e-8, 
        popsize=50, 
        init='latinhypercube',
        polish=False 
    )
    
    best_val = result_de.fun
    best_params = result_de.x
    
    # 2. Local Refinement using Nelder-Mead
    # Start from the best DE result
    result_nm = minimize(
        objective, 
        best_params, 
        method='Nelder-Mead',
        options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000}
    )
    
    if result_nm.fun < best_val:
        best_val = result_nm.fun
        best_params = result_nm.x

    # 3. Try a few other random restarts for NM in case DE got stuck in a local min
    # This is cheap and can help if the landscape is rugged
    rng = np.random.default_rng(42)
    for _ in range(10):
        p0 = [rng.uniform(0, 2*np.pi), rng.uniform(1e-4, 1e3)]
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
