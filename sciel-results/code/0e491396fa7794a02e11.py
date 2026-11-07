"""Rewritten optimizer for BBOB f23 (Katsuura) using Differential Evolution + Local Polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Hybrid Differential Evolution and Local Polish for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        calls = 0

        def wrapper(x):
            nonlocal calls
            if calls >= B:
                return np.inf
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Phase 1: Global Search with Differential Evolution
        # Budget allocation: ~400-450 for DE, ~50-100 for Local Polish
        # DE uses popsize * dim evaluations per generation.
        # popsize=10 -> 50 evals/gen.
        # maxiter=8 -> 8 generations.
        # Total ~ 50 * (1 + 8) = 450 evals.
        
        bounds = [(-5.0, 5.0)] * d
        
        try:
            de_result = differential_evolution(
                wrapper, 
                bounds, 
                maxiter=8, 
                popsize=10, 
                tol=1e-8, 
                seed=42,
                polish=False
            )
            best_x = de_result.x
            best_f = de_result.fun
        except Exception:
            # Fallback if DE fails
            best_x = np.zeros(d)
            best_f = wrapper(best_x)

        # Phase 2: Local Polish
        # Use remaining budget for Nelder-Mead.
        if calls < B - 10:
            res = minimize(
                wrapper, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        return best_x, best_f
# EVOLVE-BLOCK-END
