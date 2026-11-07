# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


class Optimizer:
    """Hybrid Global-Local Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        start_time = time.time()
        deadline = start_time + 280.0

        calls = 0

        def counting_func(x):
            nonlocal calls
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Global Search using Differential Evolution
        # Use a small population to conserve budget for local refinement
        # popsize=10, maxiter=20 is a reasonable trade-off for 5D
        try:
            result_de = differential_evolution(
                counting_func,
                bounds,
                maxiter=20,
                popsize=10,
                tol=1e-8,
                seed=42
            )
            if result_de.fun < best_f:
                update_best(result_de.x, result_de.fun)
        except Exception:
            pass

        # Phase 2: Local Refinement using L-BFGS-B
        # Start from the best global candidate
        if best_x is not None:
            if calls < budget - 10 and time.time() < deadline:
                try:
                    result_lbfgs = minimize(
                        counting_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if result_lbfgs.fun < best_f:
                        update_best(result_lbfgs.x, result_lbfgs.fun)
                except Exception:
                    pass

        # Phase 3: Secondary Local Refinement from Origin (Fallback/Safety)
        # In case DE got stuck in a local minimum far from origin, 
        # try a quick local search from origin if budget remains
        if calls < budget - 10 and time.time() < deadline:
            x_origin = np.zeros(dim)
            try:
                result_origin = minimize(
                    counting_func,
                    x_origin,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result_origin.fun < best_f:
                    update_best(result_origin.x, result_origin.fun)
            except Exception:
                pass

        # Phase 4: Final Polish with Nelder-Mead if budget allows
        if best_x is not None and calls < budget - 5 and time.time() < deadline:
            try:
                result_nm = minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result_nm.fun < best_f:
                    update_best(result_nm.x, result_nm.fun)
            except Exception:
                pass

        # Fallback if no calls were made or best is still inf
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
