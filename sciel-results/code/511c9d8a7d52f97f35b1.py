import numpy as np
import time

class Optimizer:
    """(1+1)-ES with restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls = 0

        # Parameters
        max_restarts = 100
        restart_threshold = 20
        sigma_init = 2.0
        sigma_min = 1e-4

        for _ in range(max_restarts):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break

            # Initialize new search point
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = sigma_init
            no_improve_count = 0

            # Run (1+1)-ES
            while calls < self.budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87
                    no_improve_count += 1

                if no_improve_count >= restart_threshold or sigma < sigma_min:
                    break

        if best_x is None:
            # Fallback if no valid call was made (should not happen)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
