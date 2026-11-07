"""Optimized program using grid-based adaptive search for Rastrigin optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Grid-based adaptive search for BBOB-style Rastrigin (f3).
    Uses recursive bisection to systematically narrow down the search region.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search
        # Start with 2^dim = 32 points for dim=5
        # We'll do multiple levels of refinement

        # Level 0: 2^5 = 32 points
        # Level 1: 2^5 = 32 points in the best cell
        # Level 2: 2^5 = 32 points in the best cell
        # etc.

        # Each level uses 32 evaluations
        # With budget 500, we can do about 15 levels

        # Start with the full box [-5, 5]^dim
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        # We'll do adaptive refinement
        # At each step, divide the current best region into 2^dim sub-regions
        # and evaluate the center of each sub-region

        # Track the best region
        best_lower = lower.copy()
        best_upper = upper.copy()

        # Number of levels to attempt
        max_levels = 12  # 32 * 12 = 384 evaluations for the grid
        # Plus some for final local refinement

        for level in range(max_levels):
            if time.time() > deadline - 20:
                break
            if calls_made >= budget - 50:  # Reserve 50 for local refinement
                break

            # Create 2^dim sub-cells
            # For each dimension, split into 2 halves
            # Evaluate the center of each sub-cell

            # Generate all 2^dim combinations
            n_subcells = 2 ** dim
            subcell_centers = np.zeros((n_subcells, dim))

            for i in range(n_subcells):
                for d in range(dim):
                    bit = (i >> d) & 1
                    if bit == 0:
                        subcell_centers[i, d] = (best_lower[d] + best_upper[d]) / 2 - (best_upper[d] - best_lower[d]) / 4
                    else:
                        subcell_centers[i, d] = (best_lower[d] + best_upper[d]) / 2 + (best_upper[d] - best_lower[d]) / 4

            # Evaluate all sub-cell centers
            level_best_f = np.inf
            level_best_idx = -1

            for i in range(n_subcells):
                if time.time() > deadline - 20:
                    break
                if calls_made >= budget - 50:
                    break

                x = subcell_centers[i]
                fx = counted_func(x)

                if fx < level_best_f:
                    level_best_f = fx
                    level_best_idx = i

            # Update the best region
            if level_best_idx >= 0:
                for d in range(dim):
                    bit = (level_best_idx >> d) & 1
                    mid = (best_lower[d] + best_upper[d]) / 2
                    if bit == 0:
                        best_upper[d] = mid
                    else:
                        best_lower[d] = mid

            # Check if the region is small enough to stop
            region_size = np.max(best_upper - best_lower)
            if region_size < 0.01:
                break

        # Phase 2: Local refinement around the best point found
        if calls_made < budget - 5 and time.time() < deadline - 10:
            # Use a simple local search with decreasing step sizes
            current_x = best_x.copy()
            step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]

            for step in step_sizes:
                if calls_made >= budget or time.time() > deadline - 5:
                    break

                # Try moving in each dimension ±step
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 5:
                        break

                    for direction in [-1, 1]:
                        if calls_made >= budget or time.time() > deadline - 5:
                            break

                        x_try = current_x.copy()
                        x_try[d] += direction * step
                        x_try = np.clip(x_try, -5, 5)

                        fx = counted_func(x_try)

                        if fx < best_f:
                            current_x = x_try.copy()

        # Phase 3: If we have budget left, do a final random perturbation search
        if calls_made < budget - 3 and time.time() < deadline - 3:
            rng = np.random.default_rng(123)
            n_extra = min(budget - calls_made, 30)
            for _ in range(n_extra):
                if calls_made >= budget or time.time() > deadline - 2:
                    break
                x_try = best_x + rng.normal(0, 0.1, dim)
                x_try = np.clip(x_try, -5, 5)
                counted_func(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
