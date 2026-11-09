"""Optimizer for BBOB f23 (Katsuura) on [-5,5]^5 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart CMA-ES style optimizer for Katsuura function.
    Uses a portfolio of random restarts with adaptive step sizes,
    Latin Hypercube initialization, and a deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._deadline_margin = 5.0  # seconds to spare before 330s hard limit

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 330.0 - self._deadline_margin
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Latin Hypercube Sampling for broad initial exploration
        n_lhs = min(50, self.budget // 10)
        dim = self.dim
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5.0 + 10.0 * (perm + rng.random(n_lhs)) / n_lhs

        for i in range(n_lhs):
            if check_deadline() or calls_made >= self.budget:
                break
            evaluate(lhs_points[i])

        # Phase 2: Multi-restart (1+1)-ES with adaptive sigma
        # Each restart gets a portion of the remaining budget
        remaining_budget = self.budget - calls_made
        n_restarts = min(10, max(1, remaining_budget // 50))
        budget_per_restart = max(10, remaining_budget // n_restarts)

        for r in range(n_restarts):
            if check_deadline() or calls_made >= self.budget:
                break

            # Initialize near the current best point with some perturbation
            if best_x is not None:
                x = best_x + rng.standard_normal(dim) * 1.0
            else:
                x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)

            sigma = 1.0
            sigma_min = 1e-6
            sigma_max = 5.0
            no_improve_count = 0
            max_no_improve = 20

            for step in range(budget_per_restart):
                if check_deadline() or calls_made >= self.budget:
                    break

                # Adaptive sigma based on success rate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, sigma_max)
                    no_improve_count = 0
                else:
                    sigma = max(sigma * 0.85, sigma_min)
                    no_improve_count += 1

                # Restart within this block if stuck
                if no_improve_count >= max_no_improve:
                    if best_x is not None:
                        x = best_x + rng.standard_normal(dim) * 0.5
                    else:
                        x = rng.uniform(-5, 5, dim)
                    x = np.clip(x, -5.0, 5.0)
                    fx = evaluate(x)
                    sigma = 0.5
                    no_improve_count = 0

        # Phase 3: Fine-grained local search around the best point
        if best_x is not None and not check_deadline() and calls_made < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for step in range(self.budget - calls_made):
                if check_deadline() or calls_made >= self.budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    sigma = max(sigma * 0.9, 1e-6)

        return best_x, best_f
# EVOLVE-BLOCK-END
