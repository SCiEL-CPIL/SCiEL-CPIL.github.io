import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using LHS for global basin identification
    and Nelder-Mead for local exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        best_x = None
        fbest = np.inf
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, fbest
            if fx < fbest:
                fbest = fx
                best_x = x.copy()

        # Phase 1: Global Exploration using Latin Hypercube Sampling (LHS)
        # LHS provides deterministic global coverage, which is crucial for
        # identifying the correct basin in the bi-Rastrigin landscape.
        n_lhs = 30
        if budget < 50:
            n_lhs = budget // 2
            if n_lhs < 2:
                n_lhs = 2

        # Generate LHS points
        # We use a simple deterministic LHS generation
        pts = np.zeros((n_lhs, dim))
        for i in range(dim):
            # Shuffle indices for each dimension to ensure spread
            indices = np.arange(n_lhs)
            np.random.shuffle(indices)
            # Map indices to range [-4, 4] to stay well within bounds and cover the likely minimizer location
            # The problem states minimizer lies in [-4, 4]^5
            pts[:, i] = -4.0 + (8.0 / (n_lhs - 1 if n_lhs > 1 else 1)) * indices

        # Evaluate LHS points
        for i in range(n_lhs):
            if not time_ok() or calls >= budget:
                break
            x = pts[i]
            fx = func(x)
            calls += 1
            update_best(x, fx)

        # Phase 2: Local Exploitation using Nelder-Mead
        # Start Nelder-Mead from the best point found in Phase 1.
        # Nelder-Mead is efficient for smooth, non-linear functions and
        # adapts its step size dynamically without needing gradient information.
        
        if best_x is not None and calls < budget and time_ok():
            x0 = best_x.copy()
            
            # Define a wrapper for the function to track calls and enforce deadline
            def wrapped_func(x):
                nonlocal calls, fbest, best_x
                if not time_ok() or calls >= budget:
                    return fbest  # Return current best to stop optimization gracefully
                fx = func(x)
                calls += 1
                update_best(x, fx)
                return fx

            # Nelder-Mead options
            # maxiter is set high, but we rely on the deadline and budget in wrapped_func
            # initial_simplex_size controls the starting scale
            # We scale the initial simplex based on the dimension and a reasonable step size
            initial_simplex_size = 1.0
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'initial_simplex_size': initial_simplex_size
                    }
                )
                
                # Update best if the result is better (though update_best inside wrapper should have caught it)
                if res.fun < fbest:
                    update_best(res.x, res.fun)
                    
            except Exception:
                # If Nelder-Mead fails for any reason, we continue with whatever we have
                pass

        # If we still have budget and time, we can try a few more local refinements
        # or restarts if the first NM didn't converge well.
        # However, for Lunacek, getting into the right basin is key.
        # A second NM run from a slightly perturbed best point might help if the first one got stuck.
        
        if best_x is not None and calls < budget - 10 and time_ok():
            # Perturb best point slightly
            noise = 0.5 * np.random.standard_normal(dim)
            x_perturbed = np.clip(best_x + noise, -4, 4)
            
            def wrapped_func2(x):
                nonlocal calls, fbest, best_x
                if not time_ok() or calls >= budget:
                    return fbest
                fx = func(x)
                calls += 1
                update_best(x, fx)
                return fx

            try:
                res2 = minimize(
                    wrapped_func2, 
                    x_perturbed, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'initial_simplex_size': 0.5
                    }
                )
                if res2.fun < fbest:
                    update_best(res2.x, res2.fun)
            except Exception:
                pass

        return best_x, fbest
