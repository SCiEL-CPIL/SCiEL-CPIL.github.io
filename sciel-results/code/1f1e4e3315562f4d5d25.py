"""Optimizer using Differential Evolution for global search and L-BFGS-B for local polishing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


class Optimizer:
    """Hybrid DE + L-BFGS-B optimizer for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        # Phase 1: Global Search using Differential Evolution
        # DE is a population-based global optimizer that handles box constraints natively.
        # We allocate ~40% of the budget to this phase.
        de_budget = int(budget * 0.4)
        
        if time_ok() and call_count[0] < de_budget:
            try:
                # DE requires a function that returns a float.
                # It manages its own population and evaluation count roughly.
                # We pass maxiter to control the number of generations.
                # Popsize 15 is default, so 15*5 = 75 evals per gen.
                # 40% of 500 is 200 evals. 200/75 is ~2.6 generations.
                # This is too low for DE to be effective. 
                # However, DE is stochastic. Let's try to let it run as long as possible 
                # but stop if we exceed budget.
                # Actually, DE in scipy doesn't have a hard "max_evals" stopper that is easy to interrupt mid-run without custom callback.
                # We can use a callback to stop DE if we hit the time or budget limit.
                
                def de_callback(xk, convergence):
                    if not time_ok() or call_count[0] >= de_budget:
                        return True
                    return False

                # Bounds for DE
                bounds = [(-5, 5)] * dim
                
                # Run DE
                # maxiter is an upper bound, but callback will stop it earlier if needed.
                res_de = differential_evolution(
                    wrapped_func,
                    bounds,
                    maxiter=100, # Generations
                    popsize=15,
                    tol=1e-8,
                    callback=de_callback,
                    polish=False, # We will do our own polish
                    seed=42
                )
                
                if res_de.fun < best_f:
                    best_f = res_de.fun
                    best_x = res_de.x.copy()
            except Exception:
                pass

        # If DE didn't find anything (or failed), try a few random points as fallback
        if best_x is None:
            if time_ok() and call_count[0] < budget:
                x0 = np.zeros(dim)
                f0 = wrapped_func(x0)
                best_f = f0
                best_x = x0.copy()
        
        # Phase 2: Local Refinement using L-BFGS-B
        # L-BFGS-B is efficient for smooth functions with box constraints.
        # It uses limited-memory BFGS approximation of the Hessian.
        # We use the remaining budget (60%) for this phase.
        
        if best_x is not None and time_ok() and call_count[0] < budget - 5:
            try:
                res_local = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_local.fun < best_f:
                    best_f = res_local.fun
                    best_x = res_local.x.copy()
            except Exception:
                pass
        
        # Phase 3: Final Polish with SLSQP (if time remains)
        # SLSQP can sometimes squeeze out the last bit of precision.
        if best_x is not None and time_ok() and call_count[0] < budget - 2:
            try:
                res_slsqp = minimize(
                    wrapped_func,
                    best_x,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-12}
                )
                if res_slsqp.fun < best_f:
                    best_f = res_slsqp.fun
                    best_x = res_slsqp.x.copy()
            except Exception:
                pass

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
