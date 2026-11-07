"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free BFGS-based local search with multi-start and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Track best solution found so far
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to safely evaluate and count calls
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            calls_made += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start local search using Nelder-Mead or Powell
        # We'll use a combination of random starts and local refinement.
        # Since we don't know the scale, we start with a few broad random points
        # and then refine the best one.
        
        # Initial random sampling to find a good basin
        n_initial = min(10, budget // 5)
        for _ in range(n_initial):
            x0 = rng.uniform(-5, 5, dim)
            f0 = safe_eval(x0)
            
        # If we have budget left, refine the best point found so far
        if calls_made < budget and best_x is not None:
            # Use Nelder-Mead for local optimization
            # Set maxiter to use remaining budget
            remaining = budget - calls_made
            if remaining > 0:
                try:
                    res = minimize(
                        lambda x: safe_eval(x),
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # If we still have budget, do more random starts and refine
        while calls_made < budget:
            # Random new start
            x0 = rng.uniform(-5, 5, dim)
            f0 = safe_eval(x0)
            
            if calls_made >= budget:
                break
            
            # Refine this new start if it's competitive
            if f0 < best_f * 1.5 or best_f == np.inf:
                remaining = budget - calls_made
                if remaining > 5:
                    try:
                        res = minimize(
                            lambda x: safe_eval(x),
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
