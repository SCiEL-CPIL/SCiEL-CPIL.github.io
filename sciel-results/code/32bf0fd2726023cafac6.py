"""Deterministic hierarchical grid search exploiting Katsuura fractal structure."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools


class Optimizer:
    """Hierarchical grid-based search for Katsuura function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Track best point and function value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid 3^5 = 243 evaluations
        # Covers the full domain [-5, 5]^5
        g1 = 3
        n_points = g1 ** dim
        if calls_made + n_points > budget:
            # If we can't even do the coarse grid, just do a random search?
            # But 3^5=243 < 500, so this should always pass.
            pass

        lo1 = np.full(dim, -5.0)
        hi1 = np.full(dim, 5.0)
        axes1 = [np.linspace(lo1[i], hi1[i], g1) for i in range(dim)]
        
        for idx in itertools.product(*[range(g1)] * dim):
            x = np.array([axes1[i][idx[i]] for i in range(dim)])
            evaluate(x)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Local High-Res Grid 3^5 = 243 evaluations
        # Refine around the best point found in Phase 1.
        # Box size 1.0 (radius 0.5) to match the dominant period of Katsuura.
        g2 = 3
        n_points = g2 ** dim
        
        # Only proceed if we have enough budget
        if calls_made + n_points <= budget:
            # Define box around best_x
            radius = 0.5
            lo2 = np.clip(best_x - radius, -5.0, 5.0)
            hi2 = np.clip(best_x + radius, -5.0, 5.0)
            
            # Ensure lo2 < hi2
            for i in range(dim):
                if lo2[i] >= hi2[i]:
                    lo2[i] = best_x[i] - 0.1
                    hi2[i] = best_x[i] + 0.1
            
            axes2 = [np.linspace(lo2[i], hi2[i], g2) for i in range(dim)]
            
            for idx in itertools.product(*[range(g2)] * dim):
                x = np.array([axes2[i][idx[i]] for i in range(dim)])
                evaluate(x)

        # Phase 3: Fine Local Search with remaining budget
        # Use coordinate-wise refinement
        if best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            x = best_x.copy()
            f = best_f
            
            # Adaptive step size
            step = 0.05
            
            # We will try to improve using the remaining calls.
            # Strategy: Try small perturbations.
            # Since we have very few calls (approx 14), we do a systematic coordinate scan.
            
            # 1. Scan each dimension with +/- step
            # This takes 2 * dim calls.
            # If remaining >= 2 * dim:
            if remaining >= 2 * dim:
                for _ in range(remaining // (2 * dim)):
                    improved = False
                    for i in range(dim):
                        # Try +step
                        x_plus = x.copy()
                        x_plus[i] += step
                        if np.all(x_plus >= -5.0) and np.all(x_plus <= 5.0):
                            f_plus = func(x_plus)
                            calls_made += 1
                            if f_plus < f:
                                x = x_plus
                                f = f_plus
                                best_f = f
                                best_x = x.copy()
                                improved = True
                        # Try -step
                        x_minus = x.copy()
                        x_minus[i] -= step
                        if np.all(x_minus >= -5.0) and np.all(x_minus <= 5.0):
                            f_minus = func(x_minus)
                            calls_made += 1
                            if f_minus < f:
                                x = x_minus
                                f = f_minus
                                best_f = f
                                best_x = x.copy()
                                improved = True
                    
                    if not improved:
                        step *= 0.5
                        if step < 1e-5:
                            break
            else:
                # If we have very few calls, just try a few random small steps?
                # Or just return.
                # Let's try a single best direction if possible.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
