"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start (1+1)-ES followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        rng = np.random.default_rng(0)
        budget = self.budget
        dim = self.dim

        # Phase 1: Multi-start (1+1)-ES to find a good basin
        # Use first ~70% of budget for global exploration
        es_budget = int(budget * 0.7)
        n_starts = 5
        start_budget = es_budget // n_starts

        best_x = None
        best_fx = np.inf

        for s in range(n_starts):
            if time.time() > deadline:
                break
            # Different starting points for each trial
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            sigma = 2.0
            calls = 1

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

            for _ in range(start_budget - 1):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)

        # Phase 2: Nelder-Mead polish from the best point found
        remaining_budget = budget - es_budget
        if remaining_budget > 10 and time.time() < deadline:
            try:
                # Use a bounded Nelder-Mead via penalty method
                def penalty_func(x):
                    x_clipped = np.clip(x, -5, 5)
                    val = func(x_clipped)
                    # Add penalty for going out of bounds
                    penalty = 1e6 * np.sum((x > 5)**2 + (x < -5)**2)
                    return val + penalty

                result = minimize(
                    penalty_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_fx:
                    best_x = np.clip(result.x, -5, 5)
                    best_fx = func(best_x)
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
