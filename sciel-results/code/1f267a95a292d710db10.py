"""Optimized BBOB f12 (Bent Cigar) solver with Sobol exploration + Powell refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Hybrid optimizer: Sobol global exploration followed by Powell local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s kill

        # --- Phase 1: Sobol-based global exploration ---
        # Use quasi-random points to cover the search space efficiently
        n_explore = min(100, budget // 5)
        # Generate Sobol points using scipy
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_samples = sampler.random(n_explore)
        # Map from [0,1]^dim to [-5,5]^dim
        sobol_samples = -5 + 10 * sobol_samples

        # Evaluate all exploration points
        best_x = None
        best_f = np.inf
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = sobol_samples[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # If we haven't used all budget and time allows, add more random points
        remaining_budget = budget - n_explore
        if remaining_budget > 0:
            # Add some random points for diversity
            n_random = min(remaining_budget // 2, 50)
            rng = np.random.default_rng(42)
            for i in range(n_random):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # --- Phase 2: Powell local refinement from the best point ---
        # Use Powell's method which adapts to anisotropic landscapes like Bent Cigar
        remaining = budget - n_explore - (50 if remaining_budget > 0 else 0)
        if remaining > 10 and best_x is not None:
            # Wrap func to count evaluations
            eval_count = [0]
            def wrapped_func(x):
                eval_count[0] += 1
                return func(x)

            # Powell method with maxiter controlled by remaining budget
            max_iter = min(remaining, 200)
            result = minimize(wrapped_func, best_x, method='Powell',
                              options={'maxiter': max_iter, 'maxfev': remaining})
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x

        # --- Phase 3: If still budget left, do a few more Powell refinements from top points ---
        # Re-evaluate top exploration points and refine them
        if time.time() < deadline - 30:
            # Sort exploration points by function value
            # (We didn't store all, so just try a few more from center and corners)
            additional_starts = [
                np.zeros(dim),
                np.ones(dim) * 2.0,
                -np.ones(dim) * 2.0,
            ]
            for start in additional_starts:
                if time.time() > deadline:
                    break
                # Quick local search with Powell from this start
                eval_count2 = [0]
                def wrapped_func2(x):
                    eval_count2[0] += 1
                    return func(x)
                
                # Limit evaluations per refinement
                max_eval = min(30, budget - n_explore - 50 - max_iter)
                if max_eval <= 0:
                    break
                res = minimize(wrapped_func2, start, method='Powell',
                              options={'maxiter': 20, 'maxfev': max_eval})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x

        return best_x, best_f
# EVOLVE-BLOCK-END
