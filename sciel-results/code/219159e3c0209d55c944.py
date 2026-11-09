"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and final local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        # Phase 1: Multi-start (1+1)-ES with restarts
        # We'll do several short runs and keep the best
        max_restarts = 5
        budget_per_restart = max(50, self.budget // max_restarts)

        for restart in range(max_restarts):
            if check_deadline() or calls_made >= self.budget - 50:
                break

            # Initialize: first restart at origin, others random
            if restart == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)

            fx = do_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 3.0
            min_sigma = 1e-6
            max_sigma = 10.0

            # (1+1)-ES loop for this restart
            steps_in_restart = 0
            max_steps = budget_per_restart
            while steps_in_restart < max_steps:
                if check_deadline() or calls_made >= self.budget - 50:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = do_call(y)
                steps_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, min_sigma)

        # Phase 2: Final local polish using scipy
        if not check_deadline() and calls_made < self.budget - 5:
            try:
                from scipy.optimize import minimize

                # Use a few different starting points for polish
                polish_starts = [best_x.copy()]
                # Add a couple of random points near the best
                for _ in range(2):
                    px = best_x + 0.5 * rng.standard_normal(self.dim)
                    px = np.clip(px, -5, 5)
                    polish_starts.append(px)

                for ps in polish_starts:
                    if check_deadline() or calls_made >= self.budget - 3:
                        break

                    # Count how many calls the polish will use
                    # We'll limit the polish to use remaining budget
                    remaining = self.budget - calls_made
                    if remaining < 10:
                        break

                    def objective(x):
                        nonlocal calls_made
                        if check_deadline() or calls_made >= self.budget:
                            raise ValueError("budget exceeded")
                        val = func(np.clip(x, -5, 5))
                        calls_made += 1
                        return val

                    bounds = [(-5, 5)] * self.dim
                    try:
                        res = minimize(
                            objective,
                            ps,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-12}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

            except Exception:
                pass

        # If we haven't made any calls (shouldn't happen), make one
        if calls_made == 0:
            x = np.zeros(self.dim)
            fx = do_call(x)
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
