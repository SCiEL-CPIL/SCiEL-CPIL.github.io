"""Portfolio optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and diverse initializations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse random sampling to find a good starting region
        # Use 10% of budget for exploration
        explore_budget = max(10, self.budget // 10)

        # Latin Hypercube sampling for diversity
        n_lhs = min(explore_budget, 20)
        for i in range(n_lhs):
            # LHS: one point per stratum in each dimension
            x = np.zeros(self.dim)
            for d in range(self.dim):
                lo = -5.0 + d * 10.0 / self.dim
                hi = lo + 10.0 / self.dim
                x[d] = lo + rng.random() * (hi - lo)
            eval_point(x)

        # Fill remaining explore budget with random points
        remaining_explore = explore_budget - n_lhs
        for _ in range(remaining_explore):
            x = rng.uniform(-5.0, 5.0, self.dim)
            eval_point(x)

        if best_x is None:
            # Should not happen, but fallback
            x0 = np.zeros(self.dim)
            eval_point(x0)

        # Phase 2: Multi-restart (1+1)-ES from the best point and diverse neighbors
        # Use remaining budget for local optimization
        remaining_budget = self.budget - call_count

        # Determine number of restarts
        # Each restart gets a portion of the remaining budget
        n_restarts = min(5, max(1, remaining_budget // 50))
        budget_per_restart = remaining_budget // n_restarts

        for r in range(n_restarts):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            # Starting point: best found so far, perturbed for later restarts
            if r == 0:
                x = best_x.copy()
                sigma = 1.0
            else:
                # Perturb the best point
                x = best_x + rng.normal(0, 1.5, self.dim)
                x = np.clip(x, -5.0, 5.0)
                sigma = 1.5

            fx = best_f if r == 0 else eval_point(x)

            # (1+1)-ES with adaptive sigma
            for _ in range(budget_per_restart):
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2  # increase step on success
                    if sigma > 5.0:
                        sigma = 5.0
                else:
                    sigma *= 0.85  # decrease step on failure
                    if sigma < 0.05:
                        sigma = 0.05

        # Phase 3: Fine-tune around the best point with small steps
        if call_count < self.budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            fine_tune_budget = min(self.budget - call_count, 50)
            for _ in range(fine_tune_budget):
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
