"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start L-BFGS-B with random restarts and adaptive step sizes.
    Uses multiple local optimizations from different starting points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counting_func(x):
            calls_made[0] += 1
            return func(x)

        # Strategy: Use L-BFGS-B with multiple random starts
        # Each start gets a portion of the budget
        # We'll try several different strategies:
        # 1. A few high-quality random starts with L-BFGS-B
        # 2. If budget allows, refine with smaller steps

        # First, do a quick random search to find a decent starting point
        n_random = min(50, self.budget // 10)
        for i in range(n_random):
            if calls_made[0] >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = counting_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Now use L-BFGS-B from the best point found, and also from a few other good points
        # We'll keep track of top few starting points
        candidates = []
        # Re-do random search but keep top candidates
        candidates = []
        for i in range(n_random):
            if calls_made[0] >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = counting_func(x)
            candidates.append((fx, x.copy()))
        
        # Sort by function value
        candidates.sort(key=lambda t: t[0])
        # Take top few candidates for L-BFGS-B
        top_candidates = candidates[:min(5, len(candidates))]

        # For each top candidate, run L-BFGS-B
        for fx_start, x_start in top_candidates:
            if calls_made[0] >= self.budget:
                break
            
            def wrapped_func(x):
                if calls_made[0] >= self.budget:
                    # Return a large value to stop optimization
                    return 1e10
                return counting_func(x)
            
            try:
                result = minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 50}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # If we still have budget, do a fine-grained local search around best_x
        if best_x is not None and calls_made[0] < self.budget:
            # Do a simple local search with decreasing step sizes
            step = 0.5
            for _ in range(20):
                if calls_made[0] >= self.budget:
                    break
                improved = False
                for d in range(self.dim):
                    if calls_made[0] >= self.budget:
                        break
                    for sign in [1, -1]:
                        if calls_made[0] >= self.budget:
                            break
                        x_new = best_x.copy()
                        x_new[d] += sign * step
                        x_new = np.clip(x_new, -5, 5)
                        fx_new = counting_func(x_new)
                        if fx_new < best_f:
                            best_f = fx_new
                            best_x = x_new.copy()
                            improved = True
                if not improved:
                    step *= 0.5
                if step < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
