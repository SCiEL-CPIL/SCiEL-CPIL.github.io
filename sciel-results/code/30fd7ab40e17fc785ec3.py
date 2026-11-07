"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start coordinate descent for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Scatter initial points across the box
        # Use a Sobol-like quasi-random sequence for better coverage
        # For dim=5, use a simple stratified random approach
        n_starts = 30
        points = []
        for i in range(n_starts):
            # Stratified sampling: divide each dimension into n_starts^(1/d) bins
            # For simplicity, use random but ensure spread
            x = rng.uniform(-5, 5, d)
            points.append(x)

        # Evaluate all initial points
        best_x = points[0]
        best_f = func(best_x)
        calls = 1

        # Sort initial points by fitness (best first)
        initial_fits = []
        for i, x in enumerate(points):
            if i == 0:
                initial_fits.append((best_f, x))
            else:
                if not time_ok():
                    break
                fx = func(x)
                calls += 1
                initial_fits.append((fx, x))
                if fx < best_f:
                    best_f = fx
                    best_x = x

        # Sort by fitness
        initial_fits.sort(key=lambda t: t[0])

        # Phase 2: Coordinate descent refinement from each start point
        # Allocate remaining budget to refinement
        remaining_budget = budget - calls

        # Adaptive step sizes for coordinate descent
        # Start with larger steps, shrink as we converge
        base_steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]

        for start_f, start_x in initial_fits:
            if not time_ok():
                break
            if calls >= budget:
                break

            # If this start is already worse than best, skip refinement
            # (but we might still want to try if it's close)
            if start_f > best_f + 10:
                continue

            x = start_x.copy()
            fx = start_f
            improved = True

            # Coordinate descent with adaptive steps
            for step in base_steps:
                if not time_ok():
                    break
                if calls >= budget:
                    break

                # Try each coordinate
                for dim_idx in range(d):
                    if not time_ok():
                        break
                    if calls >= budget:
                        break

                    # Try positive and negative perturbations
                    for sign in [1, -1]:
                        if not time_ok():
                            break
                        if calls >= budget:
                            break

                        y = x.copy()
                        y[dim_idx] += sign * step
                        y = np.clip(y, -5, 5)

                        fy = func(y)
                        calls += 1

                        if fy < fx:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x

                            # Try to continue in the same direction
                            # (line search lite)
                            for _ in range(3):
                                if not time_ok():
                                    break
                                if calls >= budget:
                                    break

                                y2 = x.copy()
                                y2[dim_idx] += sign * step
                                y2 = np.clip(y2, -5, 5)

                                fy2 = func(y2)
                                calls += 1

                                if fy2 < fx:
                                    x = y2
                                    fx = fy2
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x
                                else:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
