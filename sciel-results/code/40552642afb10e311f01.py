"""Optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead with coordinate-descent warm start and polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # --- Phase 1: Coordinate descent warm start ---
        # Start from origin (center of the box)
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1

        # Try a few random starts to get a better initial point
        for _ in range(3):
            if calls >= self.budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, self.dim)
            fx_try = func(x_try)
            calls += 1
            if fx_try < fx:
                x, fx = x_try, fx_try

        # Coordinate descent
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
        for ss in step_sizes:
            for _ in range(2):
                for d in range(self.dim):
                    if calls >= self.budget or time.time() > deadline:
                        return x, fx
                    for sign in [1, -1]:
                        if calls >= self.budget or time.time() > deadline:
                            return x, fx
                        x_new = x.copy()
                        x_new[d] = np.clip(x_new[d] + sign * ss, -5, 5)
                        fx_new = func(x_new)
                        calls += 1
                        if fx_new < fx:
                            x, fx = x_new, fx_new

        # --- Phase 2: Nelder-Mead optimization ---
        # Build initial simplex around current best
        simplex = [x.copy()]
        for d in range(self.dim):
            for sign in [1, -1]:
                pt = x.copy()
                pt[d] = np.clip(pt[d] + sign * 0.5, -5, 5)
                simplex.append(pt)

        def nm_func(v):
            return func(np.clip(v, -5, 5))

        try:
            res = minimize(
                nm_func,
                x0=x,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            if res.fun < fx:
                x, fx = np.clip(res.x, -5, 5), res.fun
        except Exception:
            pass

        # --- Phase 3: Polish with fine coordinate descent ---
        for ss in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]:
            if calls >= self.budget or time.time() > deadline:
                return x, fx
            for d in range(self.dim):
                if calls >= self.budget or time.time() > deadline:
                    return x, fx
                for sign in [1, -1]:
                    if calls >= self.budget or time.time() > deadline:
                        return x, fx
                    x_new = x.copy()
                    x_new[d] = np.clip(x_new[d] + sign * ss, -5, 5)
                    fx_new = func(x_new)
                    calls += 1
                    if fx_new < fx:
                        x, fx = x_new, fx_new

        return x, fx
# EVOLVE-BLOCK-END
