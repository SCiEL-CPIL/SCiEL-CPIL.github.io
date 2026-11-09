"""Rewritten optimizer for BBOB f13 (Sharp ridge) with deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.deadline = time.time() + 280.0  # safety margin before 300s

    def __call__(self, func):
        self.calls = 0
        dim = self.dim
        budget = self.budget

        # Wrap func to count calls and enforce budget
        def count_func(x):
            if self.calls >= budget:
                return 1e30
            self.calls += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            if not np.isfinite(val):
                return 1e30
            return val

        # Phase 1: Sobol quasi-random sampling for global exploration
        # Use 64 points for good early coverage
        n_sobol = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)  # [0,1]^dim
        sobol_points = sobol_points * 10.0 - 5.0  # map to [-5,5]^dim

        best_x = None
        best_f = np.inf

        for pt in sobol_points:
            if self.calls >= budget:
                break
            if time.time() > self.deadline:
                break
            f = count_func(pt)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: Local refinement using BFGS from the best point
        # BFGS is effective for smooth, non-separable functions like f13
        if self.calls < budget and time.time() < self.deadline:
            try:
                res = minimize(
                    count_func,
                    best_x,
                    method='BFGS',
                    options={'maxiter': 50, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = res.fun
            except Exception:
                pass

        # Phase 3: If budget remains, do a few more local refinements from
        # the second and third best Sobol points to diversify
        if self.calls < budget and time.time() < self.deadline:
            # Sort Sobol points by function value (we only stored the best,
            # so re-evaluate a few promising ones near the best)
            # Actually, let's just do a few random restarts + BFGS
            rng = np.random.default_rng(123)
            for _ in range(3):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                # Perturb the best point slightly
                x0 = best_x + rng.normal(0, 0.5, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                try:
                    res = minimize(
                        count_func,
                        x0,
                        method='BFGS',
                        options={'maxiter': 30, 'gtol': 1e-8, 'disp': False}
                    )
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_x = np.clip(res.x, -5.0, 5.0)
                        best_f = res.fun
                except Exception:
                    pass

        # Final check: ensure we haven't exceeded budget
        if self.calls > budget:
            self.calls = budget

        return best_x, best_f
