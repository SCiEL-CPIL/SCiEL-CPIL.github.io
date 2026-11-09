"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with origin init, multiplicative sigma adaptation, and one restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Best solution found overall
        best_x = None
        best_f = np.inf

        # Phase 1: Main (1+1)-ES from origin
        # Start at origin - good prior for BBOB instances with shifts in [-4,4]^5
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        eval_count = 1

        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 50.0

        # Track for potential restart
        restart_budget = 50  # Reserve 50 evals for restart if needed

        for i in range(budget - 1):
            if eval_count >= budget - restart_budget:
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
                sigma = min(sigma * 1.5, sigma_max)
            else:
                sigma *= 0.87

            if sigma < sigma_min:
                break

        # Phase 2: If we have remaining budget and sigma got small, do a restart
        remaining = budget - eval_count
        if remaining > 10 and sigma < 0.1:
            # Restart from a random point near best or from origin perturbation
            if rng.random() < 0.5 and best_x is not None:
                x = np.clip(best_x + rng.standard_normal(n) * 1.0, -5, 5)
            else:
                x = rng.uniform(-3, 3, n)

            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            for i in range(remaining - 1):
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
                    sigma = min(sigma * 1.5, sigma_max)
                else:
                    sigma *= 0.87

                if sigma < sigma_min:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
