"""Optimizer for BBOB-style functions using Halton-sequence coarse-to-fine search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine Halton sequence search with local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim):
        """Generate n points in [0,1)^dim using the Halton sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # bases: 2, 3, 5, 7, 11, ...
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    i_val, rem = divmod(i_val, base)
                    f /= base
                    r += rem * f
                points[i - 1, d] = r
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse-to-fine Halton search
        # Start with a moderate number of points and refine around the best
        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Coarse grid: Halton points spread over the entire box
        n_coarse = 100
        halton = self._halton_sequence(n_coarse, dim)
        # Map from [0,1)^dim to [-5,5)^dim
        coarse_points = -5.0 + 10.0 * halton

        for i in range(n_coarse):
            if calls >= budget or time.time() > deadline:
                break
            eval_point(coarse_points[i])

        # Phase 2: Local refinement around the best point found
        # Use smaller and smaller neighborhoods
        if best_x is not None:
            # Refinement stages with decreasing radius
            radii = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
            n_local = 50

            for radius in radii:
                if calls >= budget or time.time() > deadline:
                    break
                # Generate Halton points in a local box around best_x
                local_halton = self._halton_sequence(n_local, dim)
                # Map to local neighborhood
                local_points = best_x + radius * (2.0 * local_halton - 1.0)

                for j in range(n_local):
                    if calls >= budget or time.time() > deadline:
                        break
                    eval_point(local_points[j])

        # Phase 3: If we have budget left, do a finer local search
        if best_x is not None and calls < budget:
            # Even finer refinement
            for radius in [0.005, 0.002, 0.001]:
                if calls >= budget or time.time() > deadline:
                    break
                n_fine = 30
                fine_halton = self._halton_sequence(n_fine, dim)
                fine_points = best_x + radius * (2.0 * fine_halton - 1.0)

                for j in range(n_fine):
                    if calls >= budget or time.time() > deadline:
                        break
                    eval_point(fine_points[j])

        # Phase 4: Use remaining budget for a simple (1+1)-ES-like local search
        if best_x is not None and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for _ in range(budget - calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.default_rng(calls).standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
