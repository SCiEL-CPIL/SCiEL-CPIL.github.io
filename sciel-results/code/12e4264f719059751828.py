"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial step size
        sigma = 2.0

        # Track success rate for 1/5 rule
        # Using a simple counter over a sliding window
        window_size = 5
        success_window = []

        total_calls = 1  # already made one call

        while total_calls < budget:
            if time.time() >= deadline:
                break

            # Generate one candidate
            y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x = y
                fx = fy
                success_window.append(True)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                success_window.append(False)

            # Update sigma based on 1/5 success rule
            if len(success_window) >= window_size:
                # Keep only the last window_size entries
                if len(success_window) > window_size:
                    success_window = success_window[-window_size:]
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma /= 1.1

            # Clamp sigma
            sigma = max(1e-6, min(10.0, sigma))

        return best_x, best_fx
# EVOLVE-BLOCK-END
