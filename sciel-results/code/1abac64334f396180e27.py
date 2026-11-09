import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            if calls_used >= budget or check_deadline():
                return np.inf
            fx = func(np.clip(x, -5, 5))
            calls_used += 1
            return fx

        # Phase 1: Multi-start with diagonal CMA-ES-like adaptation
        # Allocate ~55% of budget to global search
        global_budget = int(budget * 0.55)
        max_restarts = 10
        budget_per_restart = max(25, global_budget // max_restarts)

        for restart in range(max_restarts):
            if check_deadline() or calls_used >= global_budget:
                break

            # Initialize: first restart at origin, others guided by progress
            if restart == 0:
                x = np.zeros(n)
            elif restart == 1:
                x = rng.uniform(-3, 3, n)
            elif restart == 2:
                # Rosenbrock minimum is near (1,1,...,1) before rotation
                x = 1.0 + rng.standard_normal(n) * 0.5
                x = np.clip(x, -5, 5)
            elif restart == 3:
                x = best_x + 1.5 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
            elif restart == 4:
                x = rng.uniform(-2, 2, n)
            else:
                # Mix: 50% near best, 30% moderate random, 20% wide random
                r = rng.random()
                if r < 0.5:
                    x = best_x + 1.0 * rng.standard_normal(n)
                elif r < 0.8:
                    x = rng.uniform(-2.5, 2.5, n)
                else:
                    x = rng.uniform(-4, 4, n)
                x = np.clip(x, -5, 5)

            fx = do_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Diagonal CMA-ES: maintain per-coordinate step sizes
            sigma = np.ones(n) * 2.0
            min_sigma = 1e-7
            max_sigma = 6.0
            # Success rate tracking for adaptive sigma
            success_count = 0
            trial_count = 0
            target_success = 0.2  # 1/5 rule for (1+1)

            steps_in_restart = 0
            max_steps = budget_per_restart
            while steps_in_restart < max_steps:
                if check_deadline() or calls_used >= global_budget:
                    break

                # Generate candidate with per-coordinate sigma
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = do_call(y)
                steps_in_restart += 1
                trial_count += 1

                if fy < fx:
                    success_count += 1
                    # Move to better point
                    x, fx = y, fy
                    # Adapt sigma: increase if success rate > target, decrease otherwise
                    # Use a smooth adaptation
                    current_rate = success_count / trial_count
                    if current_rate > target_success:
                        sigma = sigma * 1.1
                    else:
                        sigma = sigma * 0.95
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = sigma * 0.95
                    sigma = np.clip(sigma, min_sigma, max_sigma)

                # Reset success tracking periodically
                if trial_count % 20 == 0:
                    success_count = 0
                    trial_count = 0

        # Phase 2: Local polish using Nelder-Mead from multiple starting points
        # Allocate remaining budget to local refinement
        remaining_budget = budget - calls_used
        if remaining_budget > 20 and not check_deadline():
            # Use Nelder-Mead from multiple starting points near best
            polish_starts = [best_x.copy()]
            # Add nearby points at various scales for robustness
            for scale in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
                px = best_x + scale * rng.standard_normal(n)
                px = np.clip(px, -5, 5)
                polish_starts.append(px)

            # Sort by distance to best_x, try closest first
            polish_starts.sort(key=lambda p: np.linalg.norm(p - best_x))

            for ps in polish_starts:
                if check_deadline() or calls_used >= budget - 5:
                    break

                remaining = budget - calls_used
                if remaining < 15:
                    break

                def objective(x):
                    nonlocal calls_used
                    if check_deadline() or calls_used >= budget:
                        raise ValueError("budget exceeded")
                    val = func(np.clip(x, -5, 5))
                    calls_used += 1
                    return val

                try:
                    # Nelder-Mead with adaptive simplex
                    res = minimize(
                        objective,
                        ps,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining - 5, 500),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 3: Final coordinate-descent polish for high precision
        if not check_deadline() and calls_used < budget - 10:
            # Simple coordinate descent with line search
            x_cd = best_x.copy()
            f_cd = best_f
            coord_step = 0.01
            for _ in range(5):  # 5 passes over coordinates
                if check_deadline() or calls_used >= budget - 5:
                    break
                for i in range(n):
                    if check_deadline() or calls_used >= budget - 5:
                        break
                    # Try small steps in each direction
                    for delta in [coord_step, -coord_step, coord_step * 0.1, -coord_step * 0.1]:
                        if calls_used >= budget - 5:
                            break
                        x_try = x_cd.copy()
                        x_try[i] += delta
                        x_try = np.clip(x_try, -5, 5)
                        f_try = do_call(x_try)
                        if f_try < f_cd:
                            x_cd = x_try
                            f_cd = f_try
                            if f_cd < best_f:
                                best_f = f_cd
                                best_x = x_cd.copy()
                    # Shrink step if no improvement
                    coord_step *= 0.5
                    if coord_step < 1e-6:
                        break

        # If we haven't made any calls (shouldn't happen), make one
        if calls_used == 0:
            x = np.zeros(n)
            fx = do_call(x)
            best_x = x
            best_f = fx

        return best_x, best_f
