import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with step-size adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        # Phase 1: (1+1)-ES for most of the budget
        # Reserve last 50 calls for Nelder-Mead polish
        polish_budget = 50
        es_budget = budget - polish_budget

        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.8

            # Clamp sigma
            if sigma > 10.0:
                sigma = 10.0
            elif sigma < 1e-12:
                sigma = 1e-12

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            # Use Nelder-Mead starting from best point found
            # Limit iterations to stay within budget
            max_iters = remaining - 2  # Leave buffer
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iters,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_fx:
                    best_x = res.x.copy()
                    best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
