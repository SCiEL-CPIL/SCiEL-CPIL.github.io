# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-start Powell's method with deadline guard for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)

        # Track calls and best
        calls = [0]
        best_x = [None]
        best_f = [np.inf]

        def f(x):
            calls[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            if val < best_f[0]:
                best_f[0] = val
                best_x[0] = x.copy()
            return val

        # Initial best: a few random starts
        for _ in range(5):
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            x0 = rng.uniform(-5.0, 5.0, self.dim)
            f(x0)

        # Multi-start Powell's method
        # Start points: current best, plus deterministic spread points
        starts = []
        if best_x[0] is not None:
            starts.append(best_x[0].copy())

        # Deterministic spread: corners and midpoints of the box
        for _ in range(10):
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            x0 = rng.uniform(-5.0, 5.0, self.dim)
            f(x0)
            if best_x[0] is not None:
                starts.append(best_x[0].copy())

        # Add a few more spread points around the current best
        if best_x[0] is not None:
            for scale in [0.5, 1.0, 2.0]:
                if calls[0] >= self.budget or time.perf_counter() >= deadline:
                    break
                x0 = np.clip(best_x[0] + scale * rng.standard_normal(self.dim), -5.0, 5.0)
                f(x0)
                starts.append(best_x[0].copy())

        # Run Powell's method from each start
        # Powell's method uses direction set updates; scipy.optimize.powell
        # takes a function and a starting point.
        for x0 in starts:
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            try:
                res = scipy.optimize.powell(
                    f, x0,
                    maxiter=50,
                    maxfev=self.budget - calls[0],
                    disp=False,
                )
            except Exception:
                pass
            if best_x[0] is not None:
                starts.append(best_x[0].copy())

        # If budget remains, do a final round of Powell from the best point
        if calls[0] < self.budget and best_x[0] is not None and time.perf_counter() < deadline:
            try:
                res = scipy.optimize.powell(
                    f, best_x[0],
                    maxiter=100,
                    maxfev=self.budget - calls[0],
                    disp=False,
                )
            except Exception:
                pass

        # If still budget, do a Nelder-Mead polish
        if calls[0] < self.budget and best_x[0] is not None and time.perf_counter() < deadline:
            try:
                res = scipy.optimize.minimize(
                    f, best_x[0],
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': self.budget - calls[0],
                             'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
                )
            except Exception:
                pass

        # Final safety: ensure we return something
        if best_x[0] is None:
            best_x[0] = np.zeros(self.dim)
            best_f[0] = float('inf')

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
