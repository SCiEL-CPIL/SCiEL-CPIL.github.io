"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + (1+1)-ES exploitation with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        # Phase 1: Latin Hypercube Sampling for broad exploration
        # Use a fixed number of LHS points based on dimension
        n_lhs = 200
        rng = np.random.default_rng(42)

        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + rng.uniform(0, 1, n_lhs)) / n_lhs * 10.0 - 5.0

        # Evaluate LHS points
        best_x = None
        best_f = np.inf
        calls = 0

        for i in range(n_lhs):
            if calls >= budget:
                break
            x = lhs_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multiple (1+1)-ES restarts from best points
        # We'll do several restarts from the top few LHS points
        # Sort LHS points by fitness
        lhs_fitness = []
        for i in range(n_lhs):
            lhs_fitness.append((lhs_points[i], lhs_points[i]))  # placeholder, we don't have all fitnesses stored

        # Actually, let's re-evaluate top candidates or just use the best
        # Let's do (1+1)-ES from the best point found, with adaptive sigma

        # Restart strategy: try (1+1)-ES from best_x with different initial sigmas
        remaining_budget = budget - calls

        # Try a few restarts with different sigma values
        sigmas_to_try = [3.0, 1.5, 0.5, 0.2, 0.1]

        for sigma_init in sigmas_to_try:
            if remaining_budget <= 0:
                break
            if time.time() > deadline:
                break

            # Restart from best_x found so far
            x = best_x.copy()
            fx = best_f
            sigma = sigma_init

            # (1+1)-ES with 1/5 success rule
            successes = 0
            total = 0
            consecutive_failures = 0
            max_failures = 20  # restart if stuck

            for step in range(remaining_budget):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                remaining_budget -= 1
                total += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    consecutive_failures = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    consecutive_failures += 1

                # 1/5 success rule for sigma adaptation
                if total >= 10:
                    if successes / total > 1.0 / 5.0:
                        sigma *= 1.05
                    else:
                        sigma *= 1.0 / 1.05
                    successes = 0
                    total = 0

                # Restart if stuck
                if consecutive_failures >= max_failures:
                    # Try a small random perturbation
                    x = x + 0.5 * sigma * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    remaining_budget -= 1
                    consecutive_failures = 0

                # Break if sigma is too small
                if sigma < 1e-6:
                    break

            # After each restart, if we made progress, continue
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
