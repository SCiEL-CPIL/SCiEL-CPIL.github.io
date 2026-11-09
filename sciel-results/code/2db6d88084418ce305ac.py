"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: (1+1)-ES with restarts and local refinement."""

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

        # Phase 1: Global search with (1+1)-ES and restarts
        # Try multiple starting points
        start_points = [np.zeros(n)]
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, n))

        for sp in start_points:
            if time.time() > deadline - 30:
                break
            if calls_used >= budget:
                break

            x = sp.copy()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_x, best_f = x.copy(), fx

            sigma = 1.5
            no_improve = 0

            while calls_used < budget and time.time() < deadline - 30:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                # If stuck, try a larger jump
                if no_improve > 20 and sigma < 0.1:
                    y = np.clip(x + rng.uniform(-3, 3, n), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_x, best_f = x.copy(), fx
                        sigma = 1.0
                        no_improve = 0
                    else:
                        sigma *= 0.7

        # Phase 2: Local refinement using Nelder-Mead or BFGS
        if time.time() < deadline - 15 and calls_used < budget - 10:
            try:
                result = minimize(
                    lambda v: func(np.clip(v, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
