"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start optimizer with global grid + local CMA-ES-like refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit

        call_count = 0

        def call(func, x):
            nonlocal call_count
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            return fx

        # Phase 1: Global coarse grid search
        # Weierstrass f16 on BBOB is multimodal. A coarse grid helps find the basin.
        # 3^5 = 243 points. That's a good fraction of budget.
        grid_size = 3
        coords = np.linspace(-5, 5, grid_size)
        grid_points = list(np.meshgrid(*([coords] * dim), indexing='ij'))
        grid_flat = np.array([np.column_stack([g.ravel() for g in grid_points]).T])

        best_x = None
        best_f = np.inf

        for pt in grid_flat:
            if call_count >= budget or time.time() > deadline:
                break
            fx = call(func, pt)
            if fx < best_f:
                best_f = fx
                best_x = pt.copy()

        # Phase 2: Local refinement using a (1+1)-ES with adaptive step size
        # Start from the best grid point
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # smaller step for local refinement

        # Use a simple (1+1)-ES with step size adaptation
        # We have remaining budget
        remaining = budget - call_count

        # Adaptive (1+1)-ES
        # Use 1/5 rule or simple adaptation
        successes = 0
        total = 0
        sigma = 0.5

        for i in range(remaining):
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * np.random.randn(dim)
            y = x + step
            y = np.clip(y, -5.0, 5.0)

            fy = call(func, y)
            total += 1

            if fy < fx:
                successes += 1
                x = y
                fx = fy
                # Increase step size on success
                sigma *= 1.25
            else:
                # Decrease step size on failure
                sigma *= 0.8

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

            # Check if we're stuck (no improvement in last 50 steps)
            if total % 50 == 0 and successes / total < 0.05:
                # Reset to a slightly larger step or try a different approach
                sigma = max(sigma, 0.1)

        return x, fx
# EVOLVE-BLOCK-END
