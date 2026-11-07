# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start portfolio optimizer for Katsuura (f23) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_search(self, func, x0, fx0, budget, sigma0, rng, deadline):
        """(1+1)-ES local refinement with adaptive sigma."""
        x = x0.copy()
        fx = fx0
        sigma = sigma0
        calls = 0
        for _ in range(budget):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            if sigma < 1e-6:
                break
        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start exploration
        # Use Sobol-like diverse points for initial exploration
        n_starts = min(10, budget // 50)
        per_start_budget = max(30, (budget - n_starts) // n_starts)

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Generate diverse starting points
        start_points = []
        # Corner points and center
        corners = np.array([[-5]*dim, [5]*dim, [-5,5,-5,5,-5], [5,-5,5,-5,5],
                            [0]*dim, [2.5]*dim, [-2.5]*dim])
        for c in corners[:n_starts]:
            start_points.append(c + rng.uniform(-0.5, 0.5, dim))
        # Add random points
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-5, 5, dim))

        for i in range(n_starts):
            if time.time() > deadline:
                break
            x0 = np.clip(start_points[i], -5, 5)
            fx0 = func(x0)
            total_calls += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Local refinement
            remaining = per_start_budget
            if total_calls + remaining > budget:
                remaining = budget - total_calls
            x, fx, calls = self._local_search(func, x0, fx0, remaining, 1.5, rng, deadline)
            total_calls += calls
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Exploit the best candidate with remaining budget
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            x, fx, calls = self._local_search(func, best_x, best_f, remaining, 0.5, rng, deadline)
            total_calls += calls
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
