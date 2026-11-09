"""Combined optimizer: (1+1)-ES with restarts for multimodal robustness."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        calls_made = 0

        # Phase 1: Initial random sampling for diversity
        n_init = min(15, budget // 20)
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

        # Phase 2: (1+1)-ES with restarts
        max_restarts = 20
        restart_count = 0

        while calls_made < budget and restart_count < max_restarts:
            if time.time() > deadline:
                break

            # Choose starting point for this restart
            if restart_count == 0:
                x = best_x.copy()
                fx = best_f
            else:
                # Mix: sometimes perturb best, sometimes random
                if rng.random() < 0.4:
                    x = best_x + 1.5 * rng.standard_normal(dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                update_best(x, fx)

            sigma = 1.0
            no_improve = 0
            max_no_improve = 25

            while calls_made < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fy)
                    no_improve = 0
                    sigma = min(sigma * 1.4, 8.0)
                else:
                    no_improve += 1
                    sigma *= 0.85
                    if no_improve > max_no_improve:
                        break

            restart_count += 1

        # Phase 3: Local refinement around best point with small sigma
        if time.time() < deadline and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fy)
                    sigma = min(sigma * 1.15, 3.0)
                else:
                    sigma *= 0.8
                    if sigma < 0.005:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
