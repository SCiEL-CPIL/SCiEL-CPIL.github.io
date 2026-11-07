"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size and time-based deadline guard.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: use diverse initial points
        # Generate 8 diverse starting points using a quasi-random / stratified approach
        n_starts = 8
        starts = []
        for i in range(n_starts):
            # Use different seeds for diversity
            local_rng = np.random.default_rng(1000 + i * 137)
            x0 = local_rng.uniform(-4.5, 4.5, dim)
            starts.append(x0)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        # For each start, run a (1+1)-ES with adaptive sigma
        # Allocate budget across starts: more budget for earlier starts if they're promising
        # Strategy: run each start with a portion of the budget, but adaptively
        # Give each start at least some budget, but allow early termination if no progress

        # Total budget to distribute
        remaining_budget = budget

        for start_idx, x0 in enumerate(starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Allocate budget for this start
            # Early starts get more budget; later starts get less
            # But also: if we're doing well, spend more on the current best
            if start_idx < 3:
                alloc = min(remaining_budget, max(50, budget // 3))
            elif start_idx < 6:
                alloc = min(remaining_budget, max(30, budget // 6))
            else:
                alloc = min(remaining_budget, max(15, budget // 10))

            x = x0.copy()
            fx = try_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # Track success rate for adaptive sigma
            successes = 0
            steps = 0
            no_improve = 0

            for _ in range(alloc):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if fy is None:
                    break

                steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve = 0
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.85

                # Adaptive sigma based on success rate (1/5 rule inspired)
                if steps >= 10:
                    success_rate = successes / steps
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    # Reset counters periodically
                    if steps >= 50:
                        successes = 0
                        steps = 0

                # If no improvement for a while, try a larger jump
                if no_improve > 20:
                    sigma = min(sigma * 2.0, 5.0)
                    no_improve = 0

            # After each start, if we have a good result, do a local refinement
            if calls_made < budget and time.time() < deadline:
                # Local refinement around best_x with small sigma
                x_ref = best_x.copy()
                fx_ref = best_f
                sigma_ref = 0.5
                refine_budget = min(remaining_budget - calls_made, 20)
                for _ in range(refine_budget):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    y = np.clip(x_ref + sigma_ref * rng.standard_normal(dim), -5, 5)
                    fy = try_func(y)
                    if fy is None:
                        break
                    if fy < fx_ref:
                        x_ref, fx_ref = y, fy
                        sigma_ref *= 1.2
                        if fx_ref < best_f:
                            best_f = fx_ref
                            best_x = x_ref.copy()
                    else:
                        sigma_ref *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
