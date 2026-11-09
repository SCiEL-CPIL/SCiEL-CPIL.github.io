"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell local optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # Bounds
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        best_x = None
        best_f = np.inf
        
        # Strategy: Multi-start Powell
        # We split the budget into several restarts.
        # Rosenbrock is unimodal but has a narrow valley.
        # Powell is derivative-free and good for this.
        
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        remaining_budget = self.budget
        
        for i in range(n_restarts):
            if remaining_budget <= 0:
                break
            
            # Generate a random starting point
            # We can bias towards the center slightly, but uniform is fine.
            # To improve chances, we can also try the center in the first iteration
            if i == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-5.0, 5.0, size=self.dim)
            
            # Run Powell
            # maxfev controls the max function evaluations
            try:
                res = minimize(func, x0, method='Powell', bounds=bounds, 
                               options={'maxfev': budget_per_restart, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                
                remaining_budget -= res.nfev
                
            except Exception:
                # If Powell fails, assume some cost and continue
                remaining_budget -= 10
                continue
        
        # If we have remaining budget, refine the best solution found
        if best_x is not None and remaining_budget > 0:
            try:
                res_refine = minimize(func, best_x, method='Powell', bounds=bounds, 
                                      options={'maxfev': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-10})
                if np.isfinite(res_refine.fun) and res_refine.fun < best_f:
                    best_f = res_refine.fun
                    best_x = res_refine.x.copy()
            except Exception:
                pass

        # If no valid solution was found (should not happen), return a random point
        if best_x is None:
            best_x = rng.uniform(-5.0, 5.0, size=self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
