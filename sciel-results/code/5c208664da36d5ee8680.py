import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    def generate_h(params):
        phi0, mu = params
        k = np.arange(N)
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

    bounds = [(0, 2 * np.pi), (1e-6, 5e-4)]
    
    best_result = None
    best_value = float('inf')
    
    for seed in [42, 123, 456]:
        result_de = differential_evolution(
            objective, bounds, seed=seed, maxiter=200, tol=1e-5, 
            popsize=30, init='latinhypercube'
        )
        if result_de.fun < best_value:
            best_value = result_de.fun
            best_result = result_de
    
    if best_result is not None:
        result_nm = minimize(
            objective, best_result.x, method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
        )
        if result_nm.fun < best_value:
            best_value = result_nm.fun
            best_result = result_nm
    
    best_params = best_result.x
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
