"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based basin identification with Rastrigin-aware refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_refinement(self, func, x0, budget, rng, deadline, best_x, best_f):
        """Fast local search using coordinate descent and small perturbations."""
        n = self.dim
        calls = 0
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls += 1
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # Coordinate descent with adaptive step
        step = 0.5
        for _ in range(10):
            if calls >= budget or time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                for direction in [1, -1]:
                    if calls >= budget:
                        break
                    x_try = x.copy()
                    x_try[i] += direction * step
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                    if f_try < fx:
                        x = x_try
                        fx = f_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 0.01:
                    break

        # Small random perturbations around best
        for _ in range(20):
            if calls >= budget or time.time() > deadline:
                break
            x_try = best_x + rng.standard_normal(n) * 0.1
            x_try = np.clip(x_try, -5, 5)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Coarse grid sampling to map the landscape
        # Büche-Rastrigin has periodic ridges; use a grid to find promising basins
        grid_size = 3  # 3^5 = 243 points
        grid_points = []
        for i1 in range(grid_size):
            for i2 in range(grid_size):
                for i3 in range(grid_size):
                    for i4 in range(grid_size):
                        for i5 in range(grid_size):
                            x = np.array([
                                -4 + 8 * i1 / (grid_size - 1),
                                -4 + 8 * i2 / (grid_size - 1),
                                -4 + 8 * i3 / (grid_size - 1),
                                -4 + 8 * i4 / (grid_size - 1),
                                -4 + 8 * i5 / (grid_size - 1)
                            ])
                            grid_points.append(x)

        # Add some random points for diversity
        for _ in range(50):
            x = rng.uniform(-4.5, 4.5, dim)
            grid_points.append(x)

        # Evaluate grid points
        grid_results = []
        for x in grid_points:
            if calls_used >= self.budget or time.time() > deadline:
                break
            fx = func(x)
            calls_used += 1
            grid_results.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: Refine top candidates from grid
        grid_results.sort(key=lambda t: t[0])
        top_k = min(5, len(grid_results))
        
        for k in range(top_k):
            if calls_used >= self.budget or time.time() > deadline:
                break
            fx, x = grid_results[k]
            budget_here = min(40, self.budget - calls_used)
            best_x, best_f, local_calls = self._local_refinement(
                func, x, budget_here, rng, deadline, best_x, best_f
            )
            calls_used += local_calls

        # Phase 3: Final refinement from global best
        if calls_used < self.budget - 10 and time.time() < deadline:
            budget_final = min(30, self.budget - calls_used)
            best_x, best_f, final_calls = self._local_refinement(
                func, best_x, budget_final, rng, deadline, best_x, best_f
            )
            calls_used += final_calls

        return best_x, best_f
# EVOLVE-BLOCK-END
