import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = 150.0  # Safety margin before 180s limit

    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 54*x**10 + 1050*x**8 - 10080*x**6 + 42840*x**4 - 60480*x**2 + 10395
    
    def get_c12(c0, c4, c8):
        return (c0 - 12*c4 - 1680*c8) / 665280.0

    def g(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        return (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * np.exp(-x**2/2)

    def objective(params):
        c0, c4, c8 = params
        val, _ = quad(lambda x: abs(g(x, c0, c4, c8)), 0, 20, limit=50, epsabs=1e-8, epsrel=1e-8)
        return 2 * val

    # Initial guess from seed
    seed_params = np.array([-1.13, 0.04, 0.000308])
    
    # Local optimization first, as it's fast and likely to find a good local minimum
    try:
        res_local = minimize(objective, seed_params, method='BFGS', options={'maxiter': 500, 'gtol': 1e-9})
        best_params = res_local.x
        best_score = res_local.fun
    except Exception:
        best_params = seed_params
        best_score = objective(seed_params)

    # Global optimization if time permits
    if time.time() - start_time < deadline - 60:
        try:
            bounds = [(-2, 2), (-0.5, 0.5), (-0.1, 0.1)]
            res_global = differential_evolution(objective, bounds, seed=42, maxiter=50, tol=1e-5, polish=False, time_limit=deadline - 30)
            if res_global.fun < best_score:
                best_params = res_global.x
                best_score = res_global.fun
        except Exception:
            pass

    # Final polish
    if time.time() - start_time < deadline - 10:
        try:
            res_final = minimize(objective, best_params, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-10})
            if res_final.fun < best_score:
                best_params = res_final.x
                best_score = res_final.fun
        except Exception:
            pass

    return best_params
