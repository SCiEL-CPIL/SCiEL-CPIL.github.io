"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS warm-start + adaptive (1+1)-ES with jump-escape and fine polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before the 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def call(x):
            nonlocal best_x, best_f, call_count
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Latin Hypercube Sampling to locate the basin ---
        lhs_count = min(80, budget // 5)
        lhs_points = self._lhs(lhs_count, dim, rng)
        for i in range(lhs_count):
            if time.time() > deadline or call_count >= budget:
                break
            call(lhs_points[i])

        if best_x is None:
            x0 = np.zeros(dim)
            call(x0)
            best_x = x0
            best_f = np.inf
            # best_x was set by call(); fall back defensively
            if best_x is None:
                best_x = x0

        # --- Phase 2: Adaptive (1+1)-ES with jump-escape ---
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        no_improve = 0

        while call_count < budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = call(y)
            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma = min(sigma * 1.15, 3.0)
            else:
                no_improve += 1
                sigma *= 0.9
                if sigma < 0.001:
                    sigma = 0.001
                # Stuck for a while: take a larger exploratory jump
                if no_improve > 20:
                    sigma = min(sigma * 3.0, 3.0)
                    no_improve = 0

        # --- Phase 3: Fine polish with very small, shrinking steps ---
        polish_budget = min(60, budget - call_count)
        sigma_polish = 0.05
        for _ in range(polish_budget):
            if time.time() > deadline or call_count >= budget:
                break
            y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5.0, 5.0)
            fy = call(y)
            if fy < fx:
                x, fx = y, fy
                sigma_polish *= 0.9
            else:
                sigma_polish *= 0.8
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
