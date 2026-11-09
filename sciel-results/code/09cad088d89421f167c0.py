"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with LHS warm-start and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: LHS exploration to find a good starting point ---
        lhs_count = min(80, budget // 5)
        lhs_points = self._lhs(lhs_count, dim, rng)
        best_x = None
        best_f = np.inf
        for i in range(lhs_count):
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                break

        # --- Phase 2: Adaptive (1+1)-ES from the best point ---
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        calls_made = lhs_count

        for _ in range(budget - lhs_count):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                if sigma > 5.0:
                    sigma = 5.0
            else:
                sigma *= 0.85
                if sigma < 0.001:
                    sigma = 0.001

        # --- Phase 3: Polish with very small steps if time remains ---
        polish_budget = min(50, budget - calls_made)
        sigma_polish = 0.01
        for _ in range(polish_budget):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma_polish *= 0.9
            else:
                sigma_polish *= 0.7
            if sigma_polish < 1e-6:
                break

        return x, fx

    @staticmethod
    def _lhs(n, dim, rng):
        """Latin Hypercube Sampling: n points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = (perm + 0.5) / n * 10.0 - 5.0
        return points
# EVOLVE-BLOCK-END
