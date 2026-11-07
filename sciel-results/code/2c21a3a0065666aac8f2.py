"""Combined (1+1)-ES exploration with L-BFGS-B exploitation for BBOB f12 (bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES for global exploration, then L-BFGS-B for local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # Track global best
        best_x = None
        best_f = np.inf
        total_calls = 0

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: (1+1)-ES exploration
        # Reserve ~40% budget for L-BFGS-B exploitation
        es_budget = int(budget * 0.6)
        lbgfs_budget = budget - es_budget

        # Start from a random point
        x = rng.uniform(-4.5, 4.5, dim)
        fx = func(x)
        total_calls += 1
        record(x, fx)

        sigma = 2.0  # Initial step size

        # (1+1)-ES loop
        while total_calls < es_budget and time.time() < deadline:
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                record(x, fx)
                sigma *= 1.15
            else:
                sigma *= 0.90

            # Clamp sigma
            if sigma > 10.0:
                sigma = 10.0
            elif sigma < 1e-10:
                sigma = 1e-10

        # Phase 2: L-BFGS-B exploitation from the best point found by ES
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline and best_x is not None:
            lbgfs_calls = [0]

            def objective(x_vec):
                nonlocal total_calls
                if time.time() > deadline or total_calls >= budget:
                    raise Exception("budget or time exceeded")
                lbgfs_calls[0] += 1
                total_calls += 1
                val = func(x_vec)
                return val

            try:
                res = minimize(
                    objective,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': remaining,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish if budget remains
        remaining = budget - total_calls
        if remaining > 5 and time.time() < deadline and best_x is not None:
            nm_calls = [0]

            def nm_func(z):
                nonlocal total_calls
                if time.time() > deadline or total_calls >= budget:
                    raise Exception("budget or time exceeded")
                nm_calls[0] += 1
                total_calls += 1
                return func(np.clip(z, -5, 5))

            try:
                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
