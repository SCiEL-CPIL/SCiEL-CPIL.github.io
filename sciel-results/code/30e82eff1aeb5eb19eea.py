"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: multi-start (1+1)-ES with Nelder-Mead local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used, best_x, best_f
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start (1+1)-ES
        # Use several restarts to explore the landscape
        max_restarts = 8
        budget_per_restart = max(30, (budget - 50) // max_restarts)

        for restart in range(max_restarts):
            if check_deadline() or calls_used >= budget - 50:
                break

            # Initialize: first restart at origin, others random
            if restart == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-5, 5, n)

            fx = do_call(x)

            sigma = 2.0
            min_sigma = 1e-7
            max_sigma = 8.0

            # (1+1)-ES loop for this restart
            steps_in_restart = 0
            max_steps = budget_per_restart
            while steps_in_restart < max_steps:
                if check_deadline() or calls_used >= budget - 50:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = do_call(y)
                steps_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, max_sigma)
                else:
                    sigma = max(sigma * 0.85, min_sigma)

        # Phase 2: Local polish using Nelder-Mead from the best point
        if not check_deadline() and calls_used < budget - 10:
            try:
                def objective(x):
                    nonlocal calls_used
                    if check_deadline() or calls_used >= budget:
                        raise ValueError("budget exceeded")
                    val = func(np.clip(x, -5, 5))
                    calls_used += 1
                    return val

                # Nelder-Mead from the best point found
                res = minimize(
                    objective,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)

            except Exception:
                pass

        # Phase 3: If budget remains, do a few more L-BFGS-B tries
        if not check_deadline() and calls_used < budget - 5:
            try:
                def objective2(x):
                    nonlocal calls_used
                    if check_deadline() or calls_used >= budget:
                        raise ValueError("budget exceeded")
                    val = func(np.clip(x, -5, 5))
                    calls_used += 1
                    return val

                bounds = [(-5, 5)] * n
                # Try from best_x and a nearby point
                for start_pt in [best_x.copy()]:
                    if check_deadline() or calls_used >= budget - 3:
                        break
                    res = minimize(
                        objective2,
                        start_pt,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)

            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
