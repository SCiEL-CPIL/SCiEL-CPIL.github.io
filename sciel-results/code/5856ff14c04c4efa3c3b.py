"""Optimizer for BBOB-style f14 (Different powers) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma, followed by Nelder-Mead polish.
       Includes a wall-clock deadline guard to finish safely within the limit."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 290.0  # safety margin before 330s kill

        def time_ok():
            return time.perf_counter() < deadline

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # We'll do a few restarts with (1+1)-ES, then polish the best with Nelder-Mead
        # Strategy: split budget into phases
        # Phase 1: ~60% of budget for multi-start (1+1)-ES
        # Phase 2: ~40% of budget for Nelder-Mead polish on the best point

        best_x = None
        best_fx = np.inf
        eval_count = 0

        # Multi-start (1+1)-ES
        n_restarts = 5
        budget_per_restart = max(10, (int(budget * 0.6) // n_restarts))

        for restart in range(n_restarts):
            if not time_ok():
                break

            # Initialize
            if restart == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb around the current best
                if best_x is not None:
                    x = np.clip(best_x + rng.uniform(-3, 3, dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            eval_count += 1
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

            sigma = 1.5
            successes = 0
            total_steps = 0
            n_steps = 10

            for step in range(budget_per_restart):
                if not time_ok():
                    break
                if eval_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()

                if total_steps % n_steps == 0:
                    rate = successes / n_steps
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.2:
                        sigma *= 0.7
                    successes = 0

                total_steps += 1
                sigma = np.clip(sigma, 1e-6, 50.0)

        # Final polish with Nelder-Mead on the best point
        if time_ok() and best_x is not None and eval_count < budget:
            try:
                remaining = budget - eval_count
                maxfev = min(200, max(50, remaining))

                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'maxfev': maxfev,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'initial_simplex': None,
                    },
                )
                eval_count += res.nfev
                if res.fun < best_fx:
                    best_fx, best_x = res.fun, res.x.copy()
            except Exception:
                pass

        # If we still have budget and time, do another round of local search
        if time_ok() and best_x is not None and eval_count < budget:
            try:
                remaining = budget - eval_count
                if remaining > 50:
                    # Local (1+1)-ES polish
                    x = best_x.copy()
                    fx = best_fx
                    sigma = 0.5
                    successes = 0
                    total_steps = 0
                    n_steps = 8

                    for step in range(min(remaining, 150)):
                        if not time_ok():
                            break
                        if eval_count >= budget:
                            break

                        y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        eval_count += 1

                        if fy < fx:
                            x, fx = y, fy
                            successes += 1
                            if fx < best_fx:
                                best_fx, best_x = fx, x.copy()

                        if total_steps % n_steps == 0:
                            rate = successes / n_steps
                            if rate > 0.2:
                                sigma *= 1.2
                            elif rate < 0.2:
                                sigma *= 0.75
                            successes = 0

                        total_steps += 1
                        sigma = np.clip(sigma, 1e-8, 10.0)
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
