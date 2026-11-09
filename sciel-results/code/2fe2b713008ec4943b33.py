"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead + adaptive (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        eval_count = [0]

        def wrapped_func(x):
            eval_count[0] += 1
            if time.time() > deadline:
                raise TimeoutError
            return func(x)

        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start Nelder-Mead
        n_starts = 5
        maxiter_nm = 20  # ~100 evals total for NM
        
        for i in range(n_starts):
            if time.time() > deadline or eval_count[0] >= budget:
                break
            x0 = rng.uniform(-5, 5, dim)
            try:
                res = minimize(
                    wrapped_func, x0, method='Nelder-Mead',
                    options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except TimeoutError:
                break
            except Exception:
                continue

        # Fallback if NM failed completely
        if best_x is None:
            x0 = rng.uniform(-5, 5, dim)
            try:
                best_f = wrapped_func(x0)
                best_x = x0
            except TimeoutError:
                # Return zero vector as last resort
                return np.zeros(dim), 0.0

        # Phase 2: Adaptive (1+1)-ES polish
        x = best_x
        fx = best_f
        sigma = 0.5
        s = 0.0
        c = 0

        while eval_count[0] < budget:
            if time.time() > deadline:
                break
            c += 1
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            try:
                fy = wrapped_func(y)
            except TimeoutError:
                break

            if fy < fx:
                x, fx = y, fy
                s += 1.0
            else:
                s -= 1.0

            if c % dim == 0:
                target = s / (1.0 * dim)
                sigma *= np.exp(0.33 * (target - 1.0 / 5.0))
                s = 0.0
                sigma = np.clip(sigma, 1e-8, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
