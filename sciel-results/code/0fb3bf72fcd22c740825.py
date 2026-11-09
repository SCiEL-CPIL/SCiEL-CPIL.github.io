"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start LHS + adaptive (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for diverse starts
        n_starts = min(20, budget // 10)
        lhs_points = np.empty((n_starts, dim))
        for d in range(dim):
            perm = rng.permutation(n_starts)
            lhs_points[:, d] = -5.0 + 10.0 * (perm + rng.uniform(0, 1, n_starts)) / n_starts
        lhs_points = np.clip(lhs_points, -5.0, 5.0)

        # Evaluate all starts
        best_x = None
        best_f = np.inf
        eval_count = 0
        for i in range(n_starts):
            xi = lhs_points[i]
            fi = func(xi)
            eval_count += 1
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Also evaluate origin and a few random points if budget allows
        extra_starts = [np.zeros(dim), np.full(dim, 0.5), np.full(dim, -0.5)]
        for xi in extra_starts:
            if eval_count >= budget:
                break
            fi = func(xi)
            eval_count += 1
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Phase 2: Adaptive (1+1)-ES from best start with 1/5 rule
        x = best_x
        fx = best_f
        sigma = 1.5
        s = 0.0  # success counter
        c = 0    # total steps
        n_success = 0

        remaining = budget - eval_count
        for step in range(remaining):
            # Time guard
            if time.time() > deadline:
                break

            c += 1
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                s += 1.0
                n_success += 1
            else:
                s -= 1.0

            # 1/5 success rule: update sigma every 1/dim steps
            if c % dim == 0:
                target = s / (1.0 * dim)
                sigma *= np.exp(0.33 * (target - 1.0 / 5.0))
                s = 0.0
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
