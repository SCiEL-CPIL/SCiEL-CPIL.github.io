import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES for global search on Bent Cigar, with Nelder-Mead and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track calls carefully
        total_calls = [0]

        def tracked_func(x):
            if time.time() >= deadline:
                raise Exception("timeout")
            total_calls[0] += 1
            if total_calls[0] > budget:
                raise Exception("budget exceeded")
            return func(np.clip(x, -5, 5))

        # Initial point: center is a reasonable start for Bent Cigar
        x0 = np.zeros(dim)
        fx0 = tracked_func(x0)

        best_x = x0.copy()
        best_fx = fx0

        # Phase 1: CMA-ES for most of the budget
        # CMA-ES handles the anisotropy of Bent Cigar well
        polish_budget = 60
        es_budget = budget - polish_budget

        if es_budget > 10 and time.time() < deadline:
            try:
                # CMA-ES options
                # sigma0: initial step size
                # maxiter: limit iterations to stay within budget
                result = minimize(
                    tracked_func,
                    x0,
                    method='CMA',
                    options={
                        'maxiter': es_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'verbose': 0,
                        'bounds': [(-5, 5)] * dim,
                        'seed': 42,
                        'CMA_verb': 0,
                    }
                )
                if result.fun < best_fx:
                    best_x = np.clip(result.x, -5, 5)
                    best_fx = result.fun
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - total_calls[0]
        if remaining > 5 and time.time() < deadline:
            nm_calls = [0]

            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(np.clip(z, -5, 5))

            try:
                result2 = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result2.fun < best_fx:
                    best_x = np.clip(result2.x, -5, 5)
                    best_fx = result2.fun
            except Exception:
                pass

        # Phase 3: L-BFGS-B polish if budget remains
        remaining2 = budget - total_calls[0]
        if remaining2 > 3 and time.time() < deadline:
            lb_calls = [0]

            def lb_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if lb_calls[0] >= remaining2:
                    raise Exception("budget exceeded")
                lb_calls[0] += 1
                return func(np.clip(z, -5, 5))

            try:
                bounds = [(-5, 5)] * dim
                result3 = minimize(
                    lb_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': remaining2, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                if result3.fun < best_fx:
                    best_x = np.clip(result3.x, -5, 5)
                    best_fx = result3.fun
            except Exception:
                pass

        return best_x, best_fx
