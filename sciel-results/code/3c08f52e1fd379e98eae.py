"""Portfolio optimizer: diversified (1+1)-ES with coordinate refinement and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: diversified (1+1)-ES with adaptive sigma, coordinate refinement, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 1.0

        def calls_left():
            return budget - calls

        rng = np.random.default_rng(42)

        # --- Phase 1: Diversified (1+1)-ES from multiple starts ---
        # Strategy 1: Center start with large step
        x = np.zeros(dim)
        fx = func(x)
        calls += 1
        update_best(x, fx)
        sigma = 3.0

        # Use ~40% of budget for first strategy
        max_steps_1 = max(1, (budget * 40) // 100)
        for _ in range(max_steps_1):
            if not time_left() or calls >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            update_best(y, fy)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.9

        # Strategy 2: Random restart with moderate step
        if time_left() and calls < budget:
            x2 = rng.uniform(-4, 4, dim)
            fx2 = func(x2)
            calls += 1
            update_best(x2, fx2)
            sigma2 = 1.5
            max_steps_2 = max(1, (budget * 30) // 100)
            for _ in range(max_steps_2):
                if not time_left() or calls >= budget:
                    break
                y = np.clip(x2 + sigma2 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                if fy < fx2:
                    x2, fx2 = y, fy
                    sigma2 *= 1.2
                else:
                    sigma2 *= 0.9

        # Strategy 3: Local refinement around best found
        if time_left() and calls < budget:
            x3 = best_x.copy()
            fx3 = best_f
            sigma3 = 0.5
            max_steps_3 = max(1, (budget * 20) // 100)
            for _ in range(max_steps_3):
                if not time_left() or calls >= budget:
                    break
                y = np.clip(x3 + sigma3 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                if fy < fx3:
                    x3, fx3 = y, fy
                    sigma3 *= 1.1
                else:
                    sigma3 *= 0.95

        # --- Phase 2: Coordinate-wise line search refinement ---
        if time_left() and calls < budget:
            x4 = best_x.copy()
            fx4 = best_f
            for coord in range(dim):
                if not time_left() or calls >= budget:
                    break
                for direction in [1, -1]:
                    if not time_left() or calls >= budget:
                        break
                    step = 1.0
                    improved = True
                    while improved and time_left() and calls < budget:
                        improved = False
                        for trial in [step, step * 0.5, step * 2.0]:
                            if calls >= budget or not time_left():
                                break
                            trial_x = x4.copy()
                            trial_x[coord] = np.clip(x4[coord] + direction * trial, -5, 5)
                            trial_f = func(trial_x)
                            calls += 1
                            if trial_f < fx4:
                                x4, fx4 = trial_x, trial_f
                                update_best(x4, fx4)
                                improved = True
                                break
                        if improved:
                            step *= 1.5
                        else:
                            break

        # --- Phase 3: Nelder-Mead simplex refinement ---
        if time_left() and calls < budget:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-4, 'adaptive': True}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
