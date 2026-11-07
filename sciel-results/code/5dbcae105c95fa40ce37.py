"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Sobol initialization for Discus."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, use 295s for safety

        dim = self.dim
        budget = self.budget

        # Track all evaluations
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def wrapped_func(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Generate Sobol sequence for multi-start
        from scipy.stats import qmc

        # We'll use a few hundred starting points
        n_starts = 50
        # Use Sobol sampling
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Get sample points in [0,1]^dim
        sobol_points = sampler.random(n_starts)
        # Map to [-5, 5]^dim
        start_points = sobol_points * 10.0 - 5.0

        # Also add center and some random points for diversity
        extra_points = [np.zeros(dim)]
        rng = np.random.default_rng(42)
        for _ in range(10):
            extra_points.append(rng.uniform(-5, 5, dim))

        all_starts = [p for p in start_points] + extra_points

        # Shuffle starts to get good early progress
        rng.shuffle(all_starts)

        # Try each start with L-BFGS-B
        for start_x in all_starts:
            if time.time() > deadline - 5:
                break
            if self._call_count >= budget:
                break

            try:
                result = minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': 50}
                )

                if result.fun < self._best_f:
                    self._best_f = result.fun
                    self._best_x = result.x.copy()

            except Exception:
                pass

        # If we still have budget and time, do a local refinement with (1+1)-ES
        remaining_calls = budget - self._call_count
        if remaining_calls > 10 and self._best_x is not None and time.time() < deadline - 10:
            x = self._best_x.copy()
            fx = self._best_f
            sigma = 0.5

            for _ in range(min(remaining_calls - 1, 100)):
                if time.time() > deadline - 5:
                    break
                if self._call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapped_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

                if self._best_x is None or fx < self._best_f:
                    self._best_x = x.copy()
                    self._best_f = fx

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
