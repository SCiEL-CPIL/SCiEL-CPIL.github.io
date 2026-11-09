"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 1.5
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Phase 1: (1+1)-ES with 1/5 success rule
        # Use ~80% of budget for exploration
        es_budget = int(self.budget * 0.80)
        while calls < es_budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

        # Phase 2: Nelder-Mead polish with remaining budget
        remaining = self.budget - calls
        if remaining > 10 and time.time() < deadline:
            try:
                call_count = [0]
                def wrapped_func(z):
                    call_count[0] += 1
                    if call_count[0] > remaining or time.time() > deadline:
                        return best_f
                    return func(z)
                
                res = scipy_optimize_minimize(wrapped_func, best_x, 
                                               method='Nelder-Mead',
                                               options={'maxiter': min(remaining, 200), 
                                                        'xatol': 1e-8, 
                                                        'fatol': 1e-10, 
                                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f


def scipy_optimize_minimize(func, x0, method, options):
    """Wrapper for scipy.optimize.minimize to avoid import overhead at top level."""
    import scipy.optimize
    return scipy.optimize.minimize(func, x0, method=method, options=options)
# EVOLVE-BLOCK-END
