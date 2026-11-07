"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: try several initial points
        # Use a mix of center-biased and random points
        n_starts = min(8, budget // 30)  # at least 1, at most 8
        if budget >= 200:
            n_starts = 8
        elif budget >= 100:
            n_starts = 5
        else:
            n_starts = 3

        # Generate diverse starting points
        starts = []
        # Center point
        starts.append(np.zeros(dim))
        # Random points biased toward center
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf

        # Evaluate all starts
        for s in starts:
            if time.time() > deadline:
                break
            fs = func(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Now do local optimization from the best start
        # Use a (1+1)-ES with adaptive sigma and occasional restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Track stagnation
        no_improve = 0
        stagnation_limit = 20

        # Phase 1: Local optimization with adaptive step size
        calls_made = n_starts
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate perturbation
            perturbation = rng.standard_normal(dim)
            y = np.clip(x + sigma * perturbation, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
                # Increase sigma slightly on success
                sigma = min(sigma * 1.2, 4.0)
            else:
                no_improve += 1
                # Decrease sigma on failure
                sigma = max(sigma * 0.9, 0.01)

            # Anti-stagnation: if stuck, kick with larger step
            if no_improve >= stagnation_limit:
                # Try a larger random jump
                jump = rng.uniform(-3, 3, dim)
                y_jump = np.clip(x + jump, -5, 5)
                fy_jump = func(y_jump)
                calls_made += 1
                if fy_jump < fx:
                    x, fx = y_jump, fy_jump
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 2.0
                    no_improve = 0
                else:
                    # Reset to best and try again with larger sigma
                    x = best_x.copy()
                    fx = best_f
                    sigma = 2.5
                    no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
