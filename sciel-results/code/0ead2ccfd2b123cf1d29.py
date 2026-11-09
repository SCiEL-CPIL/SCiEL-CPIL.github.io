"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so


class Optimizer:
    """
    Black-box continuous optimizer for BBOB f22 (Gallagher 21 peaks).
    Uses SciPy's Differential Evolution algorithm with Nelder-Mead polishing.
    
    This approach combines a global population-based search (DE) with a local
    deterministic search (Nelder-Mead) to refine the best candidate.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Define bounds
        bounds = [(-5.0, 5.0)] * dim
        
        # Parameters for Differential Evolution
        # We want to use most of the budget for the global search,
        # leaving some for the local polish if needed, or just fitting within.
        # DE calls: popsize * maxiter
        # Polish calls: usually < 50
        # Let's allocate 90% of budget to DE and 10% to polish (implicitly).
        # Or just set maxiter such that popsize * maxiter <= budget - 50.
        
        popsize = 15
        # Reserve 50 evaluations for polish and overhead
        de_budget = max(10, budget - 50)
        maxiter = max(1, de_budget // popsize)
        
        best_x = None
        best_f = np.inf
        call_count = 0

        def tracking_func(x):
            nonlocal best_x, best_f, call_count
            call_count += 1
            if call_count > budget:
                return 1e10
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        try:
            res = so.differential_evolution(
                tracking_func, 
                bounds, 
                maxiter=maxiter, 
                popsize=popsize,
                seed=42,
                init='latinhypercube',
                tol=1e-8,
                polish=True
            )
            # The result from DE includes the polished best
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.array(res.x, copy=True)
        except Exception:
            # If an exception occurs, we rely on the tracked best_x
            pass

        if best_x is None:
            # Fallback
            best_x = np.random.uniform(-5, 5, dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
