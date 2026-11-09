"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Lattice search + (1+1)-ES refinement for Weierstrass f16.
    Phase 1: Deterministic 3^5 grid over [-5,5]^5 (243 calls) guarantees global coverage.
    Phase 2: Adaptive (1+1)-ES from best grid point for local convergence (257 calls).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Deterministic 3^5 lattice search
        # Points at -5, 0, 5 in each dimension -> 243 evaluations
        # This guarantees coverage of the entire search space
        grid_points = []
        for d in range(dim):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            if d == 0:
                                grid_points.append([-5.0, -5.0, -5.0, -5.0, -5.0])
                            elif d == 1:
                                grid_points.append([-5.0, -5.0, -5.0, -5.0, 0.0])
                            elif d == 2:
                                grid_points.append([-5.0, -5.0, -5.0, 0.0, -5.0])
                            elif d == 3:
                                grid_points.append([-5.0, -5.0, 0.0, -5.0, -5.0])
                            elif d == 4:
                                grid_points.append([-5.0, 0.0, -5.0, -5.0, -5.0])

        # Generate all 3^5 = 243 points properly
        grid_points = []
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            x = np.array([
                                -5.0 + i1 * 5.0,
                                -5.0 + i2 * 5.0,
                                -5.0 + i3 * 5.0,
                                -5.0 + i4 * 5.0,
                                -5.0 + i5 * 5.0
                            ])
                            grid_points.append(x)

        for x in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: (1+1)-ES refinement from best grid point
        x = best_x.copy() if best_x is not None else np.zeros(dim)
        fx = best_f

        sigma = 2.0  # Initial step size
        stagnation = 0
        max_stagnation = 15

        while calls < budget and time.time() < deadline:
            # Generate one offspring
            y = x + sigma * np.random.default_rng(calls).standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = evaluate(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15  # Expand on success
                stagnation = 0
            else:
                sigma *= 0.85  # Shrink on failure
                stagnation += 1

            # Clamp sigma
            sigma = np.clip(sigma, 0.001, 5.0)

            # Restart if stuck for too long
            if stagnation > max_stagnation:
                if sigma < 0.1:
                    # Reset to best point with larger sigma
                    x = best_x.copy()
                    fx = best_f
                    sigma = 2.0
                    stagnation = 0
                else:
                    # Try a larger jump
                    y = x + sigma * 3.0 * np.random.default_rng(calls + 1000).standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        stagnation = 0
                    else:
                        sigma *= 0.7
                        stagnation = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
