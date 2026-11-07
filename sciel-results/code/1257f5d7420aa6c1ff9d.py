"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start (1+1)-ES with Nelder-Mead polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        calls_made = 0
        best_f = float('inf')
        best_x = None

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start (1+1)-ES
        # Use 5 restarts, each with a fraction of the budget
        n_starts = 5
        es_budget_per_start = max(50, (total_budget - 50) // n_starts)
        polish_budget = 50

        for start_idx in range(n_starts):
            if time.time() > deadline:
                break
            if calls_made >= total_budget - polish_budget:
                break

            # Different initialization strategies
            if start_idx == 0:
                x0 = np.zeros(dim)
            elif start_idx == 1:
                x0 = rng.uniform(-2, 2, dim)
            elif start_idx == 2:
                x0 = rng.uniform(-4, 4, dim)
            elif start_idx == 3:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Perturb the current best
                if best_x is not None:
                    x0 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                else:
                    x0 = rng.uniform(-3, 3, dim)

            x = x0.copy()
            fx = safe_func(x)
            sigma = 1.5 if start_idx == 0 else 2.0

            # 1/5 success rule with smaller window for faster adaptation
            alpha_up = 1.15
            alpha_down = 1.0 / 1.15
            window = 8
            successes = 0
            local_calls = 1

            while local_calls < es_budget_per_start:
                if time.time() > deadline:
                    break
                if calls_made >= total_budget - polish_budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)

                fy = safe_func(y)
                local_calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                else:
                    pass

                if local_calls % window == 0:
                    if successes >= window // 5:
                        sigma *= alpha_up
                    else:
                        sigma *= alpha_down
                    successes = 0

        # Phase 2: Nelder-Mead polish from the best point found
        if calls_made < total_budget and best_x is not None:
            try:
                result = minimize(
                    safe_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': max(1, total_budget - calls_made),
                             'xatol': 1e-6, 'fatol': 1e-8}
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f

# EVOLVE-BLOCK-END
