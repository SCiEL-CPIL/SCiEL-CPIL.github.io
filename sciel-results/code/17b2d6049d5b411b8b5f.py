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

    # 1. Coarse Grid Search to find promising regions
    # We search over phi0 in [0, 2pi) and mu in a log-scale range
    # mu determines the "chirp rate". Larger mu -> faster frequency change.
    best_val = float('inf')
    best_params = None
    
    # Grid for phi0
    phi0s = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    # Grid for mu (log spaced from small to large)
    mus = np.logspace(-5, 3, 30)
    
    for phi0 in phi0s:
        for mu in mus:
            val = objective([phi0, mu])
            if val < best_val:
                best_val = val
                best_params = [phi0, mu]
                
    # 2. Differential Evolution around the best found region
    # We define bounds around the best_params found, or use wide bounds if grid didn't help much
    # Let's use wide bounds but initialize with the best found
    bounds = [(0, 2 * np.pi), (1e-6, 1e2)]
    
    # Run DE with multiple seeds to be safe
    de_results = []
    for seed in [42, 123, 456, 789]:
        result_de = differential_evolution(
            objective, bounds, seed=seed, maxiter=150, tol=1e-6, 
            popsize=20, init='latinhypercube'
        )
        de_results.append(result_de)
        
    best_de = min(de_results, key=lambda r: r.fun)
    
    if best_de.fun < best_val:
        best_val = best_de.fun
        best_params = best_de.x

    # 3. Nelder-Mead Refinement
    result_nm = minimize(
        objective, best_params, method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000}
    )
    
    if result_nm.fun < best_val:
        best_val = result_nm.fun
        best_params = result_nm.x

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
