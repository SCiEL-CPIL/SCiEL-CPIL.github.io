"""Optimizer for BBOB-style f14 (Different powers) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with 1/5 success rule, adaptive sigma, and a final Nelder-Mead polish.
       Includes a wall-clock deadline guard to finish safely within the limit."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 290.0  # safety margin before 330s kill

        def time_ok():
            return time.perf_counter() < deadline

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        # (1+1)-ES with 1/5 success rule
        sigma = 2.0
        successes = 0
        total_steps = 0
        n_steps = 10  # steps per sigma adaptation

        for i in range(budget - 1):
            if not time_ok():
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            total_steps += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                pass

            if total_steps % n_steps == 0:
                rate = successes / n_steps
                if rate > 0.2:
                    sigma *= 1.3
                elif rate < 0.2:
                    sigma *= 0.7
                successes = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 50.0)

        # Final polish with Nelder-Mead if time permits
        if time_ok() and best_fx > -1e10:
            try:
                from scipy.optimize import minimize

                # Use Nelder-Mead starting from best_x
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': 200,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                    },
                )
                if res.fun < best_fx:
                    best_x, best_fx = res.x, res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
