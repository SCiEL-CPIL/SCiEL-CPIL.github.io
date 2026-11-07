"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: detect gradient sign, jump to corner, then refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1

        # Phase 1: Estimate gradient direction via forward differences from origin
        # Use a moderate step size to be robust to the hidden shift/rotation
        h = 4.0  # large step to clearly detect direction
        x0 = np.zeros(dim)
        f0 = fx_best  # already evaluated at origin
        grad_signs = np.zeros(dim)
        
        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            e = np.zeros(dim)
            e[i] = h
            xp = np.clip(x0 + e, -5, 5)
            fp = func(xp)
            calls += 1
            # If f(x0 + h*e_i) < f(x0), gradient component i is negative -> go positive
            # If f(x0 + h*e_i) > f(x0), gradient component i is positive -> go negative
            if fp < f0:
                grad_signs[i] = -1.0  # decreasing in +e_i direction, so min is at +5
            else:
                grad_signs[i] = 1.0   # increasing in +e_i direction, so min is at -5
        
        # Phase 2: Jump to the corner predicted by gradient signs
        x_corner = np.where(grad_signs > 0, -5.0, 5.0)
        if not np.array_equal(x_corner, x_best):
            fx_corner = func(x_corner)
            calls += 1
            if fx_corner < fx_best:
                x_best = x_corner.copy()
                fx_best = fx_corner

        # Phase 3: Local refinement around best point using (1+1)-ES
        x = x_best.copy()
        fx = fx_best
        sigma = 0.5
        
        while calls < budget and time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                if fx < fx_best:
                    x_best, fx_best = x.copy(), fx
                sigma *= 1.3
            else:
                sigma *= 0.8
            if sigma < 1e-6:
                break
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
