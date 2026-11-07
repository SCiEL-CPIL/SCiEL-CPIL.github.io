"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with progressive step-size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            calls_used += 1
            return func(x)

        # Define restart strategy: (initial_x, sigma_start, sigma_end, num_steps)
        # Phase 1: Broad exploration from origin
        # Phase 2: Exploration from random points
        # Phase 3: Local refinement around best found

        restarts = [
            # (initial_point, sigma_start, sigma_end, fraction_of_budget)
            (np.zeros(dim), 3.0, 0.5, 0.30),       # Start at origin, explore broadly
            (np.array([2.0, -2.0, 2.0, -2.0, 2.0]), 2.0, 0.3, 0.20),  # Random-ish start
            (np.array([-3.0, 3.0, -1.0, 1.0, -3.0]), 2.0, 0.2, 0.20),  # Another random
            (np.array([1.0, 1.0, -1.0, -1.0, 1.0]), 1.0, 0.1, 0.15),   # Closer to center
            (None, 0.5, 0.01, 0.15),               # Local refinement around best
        ]

        rng = np.random.default_rng(42)

        for restart_idx, (init_x, sigma_start, sigma_end, frac) in enumerate(restarts):
            if calls_used >= budget or time.time() > deadline:
                break

            # Determine number of steps for this restart
            steps = int(budget * frac)
            steps = min(steps, budget - calls_used)
            if steps <= 0:
                break

            # Initialize
            if init_x is None:
                x = best_x.copy()
            else:
                x = init_x.copy()

            fx = safe_call(x)
            if fx is None:
                break
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # (1+1)-ES with linearly decreasing sigma
            # Use 1/5 success rule: if success rate ~20%, keep sigma; else adjust
            # Here we use a simpler approach: sigma decreases linearly over the restart
            sigma = sigma_start

            for step in range(steps):
                if calls_used >= budget or time.time() > deadline:
                    break

                # Linearly interpolate sigma
                progress = step / max(steps - 1, 1)
                sigma = sigma_start + (sigma_end - sigma_start) * progress

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = safe_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    if fy < best_f:
                        best_f, best_x = fy, y.copy()

        # Final: use remaining budget for very fine local search around best
        if calls_used < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            remaining = budget - calls_used
            for _ in range(remaining):
                if calls_used >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    best_f, best_x = fy, y.copy()
                    # Slightly increase sigma on success for exploration
                    sigma *= 1.1
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
