"""Rewritten optimizer for BBOB f8 (Rosenbrock) with time guard."""
import time
import numpy as np


class Optimizer:
    """Adaptive (1+1)-ES with Nelder-Mead polish, time-guarded."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        calls = 0
        dim = self.dim

        # Wrap func to count calls and clip
        def f(x):
            nonlocal calls
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return float(val)

        # Phase 1: Adaptive (1+1)-ES
        rng = np.random.default_rng(42)
        x = rng.uniform(-4.0, 4.0, dim)
        fx = f(x)
        sigma = 1.0
        best_x = x.copy()
        best_f = fx

        # 1/5 rule parameters
        sigma_min = 1e-10
        sigma_max = 10.0

        while calls < self.budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = f(y)
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = min(sigma * 1.3, sigma_max)
            else:
                sigma = max(sigma * 0.8, sigma_min)

        # Phase 2: Nelder-Mead local polish from best point
        if calls < self.budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                def nm_func(xv):
                    if time.time() > deadline:
                        raise TimeoutError()
                    return f(xv)

                res = minimize(
                    nm_func,
                    best_x,
                    method="Nelder-Mead",
                    options={
                        "maxiter": max(1, (self.budget - calls) // (dim + 1)),
                        "xatol": 1e-6,
                        "fatol": 1e-8,
                        "adaptive": True,
                    },
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: If time and budget remain, do a few more ES steps
        if calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(20, self.budget - calls)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    sigma = max(sigma * 0.85, sigma_min)

        return best_x, best_f
