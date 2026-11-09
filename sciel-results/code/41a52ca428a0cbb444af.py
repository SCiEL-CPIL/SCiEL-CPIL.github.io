"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5.0, 5.0)
            calls_used += 1
            return float(func(x))

        def check_deadline():
            return time.time() > deadline or calls_used >= self.budget

        # Phase 1: Broad exploration with multiple random starts
        n_explore = 15
        for i in range(n_explore):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)

        # Phase 2: Local refinement from best point using (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 5.0

        while not check_deadline():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = safe_func(y)

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, sigma_max)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma = max(sigma * 0.8, sigma_min)

            # If step size becomes too small, restart from a perturbed best
            if sigma < 0.01:
                if check_deadline():
                    break
                x = np.clip(best_x + rng.uniform(-1, 1, self.dim), -5, 5)
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 0.5

        # Phase 3: If still have budget, try a few more restarts from diverse points
        remaining_budget = self.budget - calls_used
        if remaining_budget > 30:
            n_restarts = min(5, remaining_budget // 40)
            for r in range(n_restarts):
                if check_deadline():
                    break

                # Initialize near best_x but with some perturbation
                x = np.clip(best_x + rng.uniform(-2, 2, self.dim), -5, 5)
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                sigma = 0.3
                while not check_deadline():
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = safe_func(y)

                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.15, sigma_max)
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma = max(sigma * 0.85, sigma_min)

                    if sigma < 0.01:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
