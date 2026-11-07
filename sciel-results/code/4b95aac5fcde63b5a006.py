"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with coordinate-descent polish and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 2.0
        calls_used = 1
        budget = self.budget

        # Phase 1: (1+1)-ES with adaptive sigma
        while calls_used < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= 1.3
            else:
                sigma *= 0.85

            # If sigma gets very small, switch to polish
            if sigma < 0.01:
                break

        # Phase 2: Coordinate descent polish
        # Multiple passes with decreasing step sizes
        for pass_num in range(3):
            if time.time() > deadline or calls_used >= budget:
                break
            step_sizes = [0.5, 0.1, 0.01, 0.001, 0.0001]
            for step in step_sizes:
                if time.time() > deadline or calls_used >= budget:
                    break
                for _ in range(dim):  # one sweep per step size
                    if time.time() > deadline or calls_used >= budget:
                        break
                    for i in range(dim):
                        if time.time() > deadline or calls_used >= budget:
                            break
                        # Try positive and negative steps
                        for direction in [1, -1]:
                            if time.time() > deadline or calls_used >= budget:
                                break
                            y = x.copy()
                            y[i] = np.clip(y[i] + direction * step, -5, 5)
                            fy = func(y)
                            calls_used += 1
                            if fy < fx:
                                x, fx = y, fy
                                if fx < best_fx:
                                    best_x, best_fx = x.copy(), fx
                                # Keep going in same direction
                            else:
                                break

        # Phase 3: Small random perturbations around best found
        if time.time() < deadline and calls_used < budget:
            polish_sigma = 0.01
            for _ in range(min(50, budget - calls_used)):
                if time.time() > deadline or calls_used >= budget:
                    break
                y = np.clip(best_x + polish_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_fx:
                    best_x, best_fx = y, fy

        return best_x, best_fx
# EVOLVE-BLOCK-END
