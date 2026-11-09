"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) with restarts and adaptive step size."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multiple restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Number of restarts: we'll use the budget across multiple runs
        # Each restart gets a fraction of the budget
        num_restarts = 5
        budget_per_restart = max(10, self.budget // num_restarts)

        for restart in range(num_restarts):
            if total_calls >= self.budget:
                break
            if time.time() > deadline - 10:
                break

            # Initialize with random point in the box
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            total_calls += 1

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Adaptive step size
            sigma = 1.0
            min_sigma = 1e-6
            max_sigma = 5.0
            no_improve = 0
            max_no_improve = 20 + self.dim

            calls_this_restart = 1
            max_calls_this_restart = self.budget - total_calls

            while calls_this_restart < max_calls_this_restart:
                if time.time() > deadline - 5:
                    break
                if total_calls >= self.budget:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                calls_this_restart += 1

                if fy < fx:
                    # Accept
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma = min(max_sigma, sigma * 1.2)
                    no_improve = 0
                else:
                    sigma = max(min_sigma, sigma * 0.9)
                    no_improve += 1

                # Restart if stuck for too long
                if no_improve >= max_no_improve:
                    break

        if best_x is None:
            # Fallback: return center if no valid result
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            total_calls += 1

        return best_x, best_fx
