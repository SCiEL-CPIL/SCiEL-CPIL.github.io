"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive sigma, deadline guard, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315  # Safety margin for 330s limit
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        budget = self.budget
        dim = self.dim

        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.92
            if sigma < 0.01:
                sigma = 0.01

        # Nelder-Mead polish pass with remaining budget
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline - 5:
            try:
                def nm_func(v):
                    return func(np.clip(v, -5, 5))
                
                res = minimize(
                    nm_func, x, method='Nelder-Mead',
                    options={'maxiter': min(remaining, 200), 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < fx:
                    x, fx = res.x, res.fun
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
