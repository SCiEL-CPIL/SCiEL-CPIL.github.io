"""Optimized BBOB f14 solver with (1+1)-ES and 1/5 success rule."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Phase 1: Small initial exploration to find a decent starting point
        # This combines the 'broad exploration' structural idea of the parent
        # with the efficient search of the inspiration.
        n_initial = 5
        best_x = None
        best_f = np.inf
        
        for i in range(n_initial):
            if time.time() > deadline:
                break
            # Mix of center and random to cover likely shift locations
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
             # Fallback if no calls were made (should not happen)
             best_x = np.zeros(dim)
             best_f = func(best_x)

        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        
        calls_made = n_initial

        # Phase 2: (1+1)-ES with 1/5 success rule
        # This is the 'refinement machinery' from the inspiration (score 1.0)
        while calls_made < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87

            # Safety check for sigma becoming too small or large
            if sigma < 1e-6:
                break
            if sigma > 100:
                sigma = 100

        return x, fx
# EVOLVE-BLOCK-END
