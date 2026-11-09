"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-restart (1+1)-ES
        best_x = None
        best_f = np.inf
        calls_used = 0

        # Number of restarts: we'll use the budget split across restarts
        # First restart gets a larger chunk to establish a good baseline
        # Subsequent restarts are shorter for exploration
        n_restarts = 10
        # Budget allocation: first restart gets 40%, rest share 60%
        first_budget = int(budget * 0.4)
        remaining = budget - first_budget
        restart_budgets = [first_budget] + [remaining // (n_restarts - 1)] * (n_restarts - 1)
        # Adjust last to use all budget
        restart_budgets[-1] += remaining - (remaining // (n_restarts - 1)) * (n_restarts - 1)

        for r in range(n_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            rb = restart_budgets[r]
            if r == 0:
                # First restart: start from center or a random point
                x = np.zeros(dim)
            else:
                # Subsequent restarts: perturb the best found so far
                if best_x is not None:
                    x = best_x + 1.5 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            for i in range(rb - 1):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    sigma = max(sigma, 1e-6)

        # Final polish: if we have budget left, do a fine-grained local search
        remaining_calls = budget - calls_used
        if remaining_calls > 5 and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for i in range(remaining_calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    sigma = max(sigma, 1e-8)

        return best_x, best_f
# EVOLVE-BLOCK-END
