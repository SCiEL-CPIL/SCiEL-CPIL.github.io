"""Optimizer for BBOB-style f9 (rotated Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution population search with adaptive sigma and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under the 300s wall-clock limit

        def timed(fx):
            return fx

        rng = np.random.default_rng(12345)

        # Initial population
        N = 20
        x = rng.uniform(-5.0, 5.0, size=(N, dim))
        fvals = np.array([float(func(xi)) for xi in x])
        calls = N

        best_idx = int(np.argmin(fvals))
        best_x = x[best_idx].copy()
        best_f = float(fvals[best_idx])
        F = 0.8
        CR = 0.9
        sigma = 1.5

        for _ in range(2000000):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Adaptive sigma: keep the population spread roughly inside the box
            spread = float(np.mean(np.std(x, axis=0)))
            if spread > 0.01:
                sigma = 0.9 * spread
            if sigma < 0.05:
                sigma = 0.05
            if sigma > 4.0:
                sigma = 4.0

            # Restart: if the population has collapsed, re-seed around the best
            if spread < 0.15:
                x = np.clip(best_x[None, :] + sigma * rng.standard_normal((N, dim)), -5.0, 5.0)
                fvals = np.array([float(func(xi)) for xi in x])
                calls += N
                if calls > budget:
                    break
                best_idx = int(np.argmin(fvals))
                best_x = x[best_idx].copy()
                best_f = float(fvals[best_idx])
                continue

            # Generate one offspring
            i = int(rng.integers(N))
            a = int(rng.integers(N))
            b = int(rng.integers(N))
            while a == i or b == i or a == b:
                a = int(rng.integers(N))
                b = int(rng.integers(N))
            v = x[a] + F * (x[b] - x[a]) + sigma * rng.standard_normal(dim)
            y = np.zeros(dim)
            for j in range(dim):
                if rng.random() < CR:
                    y[j] = v[j]
                else:
                    y[j] = x[i][j]
            y = np.clip(y, -5.0, 5.0)

            fy = float(func(y))
            calls += 1
            if fy < fvals[i]:
                x[i] = y
                fvals[i] = fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
