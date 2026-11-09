"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with origin init, multiplicative sigma adaptation, and multiple restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Best solution found overall
        best_x = None
        best_f = np.inf
        eval_count = 0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Main (1+1)-ES from origin
        x = np.zeros(n)
        fx = func(x)
        try_update(x, fx)

        sigma_min = 1e-10
        sigma_max = 50.0

        # Multiple restarts with adaptive thresholds
        max_restarts = 4
        for restart in range(max_restarts):
            # Run (1+1)-ES with improved sigma adaptation
            sigma = 2.5 if restart == 0 else 1.5
            no_improve = 0
            max_no_improve = 60 + restart * 20

            for i in range(budget - eval_count):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # More aggressive sigma increase on success
                    sigma = min(sigma * 1.5, sigma_max)
                    no_improve = 0
                else:
                    # More conservative decrease on failure
                    sigma *= 0.75
                    no_improve += 1

                if sigma < sigma_min or no_improve >= max_no_improve:
                    break

            # If we used all budget or time is up, stop
            if eval_count >= budget or time.time() > deadline - 5.0:
                break

            # Restart strategy: prioritize best solution with decreasing perturbation
            remaining = budget - eval_count
            if remaining < 10:
                break

            if best_x is not None:
                # Perturb best solution with decreasing scale
                perturb_scale = max(0.3, 1.5 / (restart + 1))
                x = np.clip(best_x + rng.standard_normal(n) * perturb_scale, -5, 5)
            else:
                x = np.zeros(n)

            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Final high-precision polish around best solution
        if best_x is not None and eval_count < budget and time.time() < deadline - 3.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_no_improve = 0
            max_polish_no_improve = 20

            while eval_count < budget and time.time() < deadline - 3.0 and polish_no_improve < max_polish_no_improve:
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = max(sigma * 0.8, 1e-6)
                    polish_no_improve = 0
                else:
                    sigma *= 0.5
                    polish_no_improve += 1

                if sigma < 1e-7:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
