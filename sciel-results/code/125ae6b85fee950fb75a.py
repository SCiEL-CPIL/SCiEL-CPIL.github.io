"""Rewritten program using a structured multi-phase continuous optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase continuous optimizer with random restarts and adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _local_search(self, func, x0, fx0, budget, rng, sigma0=1.0):
        """(1+1)-ES local search with adaptive sigma."""
        x = x0.copy()
        fx = fx0
        sigma = sigma0
        n = self.dim
        for _ in range(budget):
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.3, 5.0)
            else:
                sigma *= 0.85
        return x, fx

    def _coarse_search(self, func, budget, rng):
        """Coarse random search to find promising regions."""
        n = self.dim
        best_x = rng.uniform(-5, 5, n)
        best_f = func(best_x)
        used = 1
        # Random points
        while used < budget:
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            used += 1
            if fx < best_f:
                best_x, best_f = x, fx
        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Phase 1: Coarse random search (20% of budget)
        phase1_budget = max(10, budget // 5)
        if time.time() < deadline:
            x1, f1 = self._coarse_search(func, phase1_budget, rng)
            if f1 < best_f:
                best_f, best_x = f1, x1

        # Phase 2: Local refinement from best point (30% of budget)
        phase2_budget = max(10, budget * 3 // 10)
        if time.time() < deadline and best_x is not None:
            x2, f2 = self._local_search(func, best_x, best_f, phase2_budget, rng, sigma0=1.5)
            if f2 < best_f:
                best_f, best_x = f2, x2

        # Phase 3: Multiple random restarts with local search (remaining budget)
        remaining = budget - phase1_budget - phase2_budget
        n_restarts = max(1, remaining // 40)
        per_restart = max(10, remaining // n_restarts)

        for i in range(n_restarts):
            if time.time() >= deadline:
                break
            # New random starting point
            x_start = rng.uniform(-5, 5, n)
            fx_start = func(x_start)
            if fx_start < best_f:
                best_f, best_x = fx_start, x_start.copy()
            # Local search from this point
            x_r, f_r = self._local_search(func, x_start, fx_start, per_restart, rng, sigma0=2.0)
            if f_r < best_f:
                best_f, best_x = f_r, x_r

        # Final polish: fine local search from best point
        if time.time() < deadline - 5.0 and best_x is not None:
            polish_budget = max(10, min(50, int((deadline - time.time()) * 10)))
            x_final, f_final = self._local_search(func, best_x, best_f, polish_budget, rng, sigma0=0.5)
            if f_final < best_f:
                best_f, best_x = f_final, x_final

        return best_x, best_f
# EVOLVE-BLOCK-END
