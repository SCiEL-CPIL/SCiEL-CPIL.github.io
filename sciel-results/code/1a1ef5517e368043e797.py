"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with Nelder-Mead polish for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Multi-start strategy: try several starting points
        # f6 (Attractive sector) has a narrow basin, so we need good initial exploration
        n_starts = 4
        calls_per_start = max(10, (budget - 20) // n_starts)  # reserve 20 for final polish

        best_x = None
        best_fx = np.inf
        total_calls = 0

        for s in range(n_starts):
            if total_calls >= budget - 20:
                break
            if time.time() > deadline:
                break

            # Diversify starting points
            if s == 0:
                x = rng.uniform(-3, 3, dim)
            elif s == 1:
                x = rng.uniform(-1, 1, dim)
            elif s == 2:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            total_calls += 1

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # (1+1)-ES from this starting point
            sigma = 2.0 if s == 0 else 1.0
            min_sigma = 0.001
            max_sigma = 5.0
            es_calls = 0
            es_budget = calls_per_start

            while es_calls < es_budget and total_calls < budget - 20:
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                es_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.18, max_sigma)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, min_sigma)

        # Nelder-Mead polish on the best point found
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 15:
                try:
                    from scipy.optimize import minimize
                    call_count = [0]
                    best_fx_val = [best_fx]

                    def wrapped_func(z):
                        call_count[0] += 1
                        if call_count[0] > remaining:
                            return best_fx_val[0]
                        val = func(np.clip(z, -5, 5))
                        if val < best_fx_val[0]:
                            best_fx_val[0] = val
                        return val

                    # First polish: moderate tolerance
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': min(300, remaining), 'xatol': 1e-8,
                                            'fatol': 1e-10, 'adaptive': True})
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
                        total_calls += call_count[0]
                        remaining = budget - total_calls

                    # Second polish: very tight tolerance if budget remains
                    if remaining > 10:
                        call_count[0] = 0
                        best_fx_val[0] = best_fx
                        res2 = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                        options={'maxiter': min(500, remaining), 'xatol': 1e-12,
                                                 'fatol': 1e-14, 'adaptive': True})
                        if res2.fun < best_fx:
                            best_x = np.clip(res2.x, -5, 5)
                            best_fx = res2.fun
                        total_calls += call_count[0]
                except Exception:
                    pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
