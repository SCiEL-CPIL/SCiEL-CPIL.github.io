"""Rewritten optimizer for BBOB f20 (Schwefel) with coordinate-descent hybrid."""
import numpy as np
import time


class Optimizer:
    """Coordinate-descent hybrid with adaptive per-dimension step sizes,
    alternating deterministic axis line searches and random perturbations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initialize at box center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per dimension
        sigma = np.full(dim, 2.0)

        # Track success counts for 1/5 rule per dimension
        successes = np.zeros(dim)
        total_steps = 0

        def elapsed_ok():
            return time.time() < deadline

        # Phase 1: Coordinate descent with adaptive steps
        while total_steps < budget and elapsed_ok():
            # Pick a dimension to optimize (cycling with some randomness)
            if total_steps % dim == 0:
                # Full cycle through dimensions
                dim_order = list(range(dim))
                rng.shuffle(dim_order)
            else:
                dim_order = [total_steps % dim]

            for d in dim_order:
                if total_steps >= budget or not elapsed_ok():
                    break

                # Line search along dimension d
                step = sigma[d]
                improved = False

                # Try positive and negative directions
                for direction in [1, -1]:
                    if total_steps >= budget or not elapsed_ok():
                        break

                    y = x.copy()
                    y[d] = np.clip(x[d] + direction * step, -5, 5)
                    fy = func(y)
                    total_steps += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        improved = True
                        successes[d] += 1

                        # Update best
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx

                        # Increase step size on success
                        sigma[d] = min(sigma[d] * 1.3, 4.0)
                        break  # Move to next dimension after improvement

                # If no improvement, decrease step size
                if not improved:
                    sigma[d] = max(sigma[d] * 0.7, 0.01)

            # Occasionally add a random perturbation to escape local minima
            if total_steps % 50 == 0 and total_steps < budget and elapsed_ok():
                perturbation = rng.standard_normal(dim) * 0.5
                y = np.clip(x + perturbation, -5, 5)
                fy = func(y)
                total_steps += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx

            # Check if we've converged (all step sizes very small)
            if np.all(sigma < 0.05):
                # Try a larger random restart
                if total_steps + dim < budget and elapsed_ok():
                    y = np.clip(x + rng.standard_normal(dim) * 1.0, -5, 5)
                    fy = func(y)
                    total_steps += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma = np.full(dim, 1.0)
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                else:
                    break

        # Phase 2: Fine-tuning with smaller steps if budget remains
        while total_steps < budget and elapsed_ok():
            if np.all(sigma < 0.02):
                break

            d = total_steps % dim
            step = sigma[d] * 0.5  # Smaller step for fine-tuning

            improved = False
            for direction in [1, -1]:
                if total_steps >= budget or not elapsed_ok():
                    break

                y = x.copy()
                y[d] = np.clip(x[d] + direction * step, -5, 5)
                fy = func(y)
                total_steps += 1

                if fy < fx:
                    x = y
                    fx = fy
                    improved = True
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                    break

            if not improved:
                sigma[d] = max(sigma[d] * 0.8, 0.005)

        return best_x, best_fx
