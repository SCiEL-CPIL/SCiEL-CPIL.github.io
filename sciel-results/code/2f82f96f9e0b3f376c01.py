"""Combined (1+1)-ES with CMA-ES-inspired adaptation and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, coordinate-wise scaling, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290

        rng = np.random.default_rng(42)

        # Initialize at origin (center of [-5,5]^5)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Adaptive step sizes per dimension
        sigma = np.full(dim, 1.5)

        # Track success rate for global sigma adaptation
        successes = 0
        window = 10
        success_count = 0

        # Coordinate-wise scaling (diagonal covariance approximation)
        # This helps with rotated/misaligned functions like Rastrigin rotated
        diag = np.ones(dim)

        # Phase-based strategy:
        # Phase 1: Global search with larger steps (first 30% of budget)
        # Phase 2: Local refinement with smaller steps (next 40%)
        # Phase 3: Fine-tuning (last 30%)

        phase1_end = int(budget * 0.3)
        phase2_end = int(budget * 0.7)

        for step in range(1, budget):
            if time.time() >= deadline - 5:
                break

            # Determine current phase multiplier
            if step < phase1_end:
                phase_mult = 1.0
            elif step < phase2_end:
                phase_mult = 0.5
            else:
                phase_mult = 0.15

            # Generate candidate
            noise = rng.standard_normal(dim) * sigma * diag * phase_mult
            y = np.clip(x + noise, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                successes += 1
                success_count += 1

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                # Adaptive sigma: increase on success
                # Use 1/5 rule variant: if success rate > 0.3, increase
                # Otherwise, keep or slightly decrease
                if success_count >= window:
                    rate = successes / success_count
                    if rate > 0.3:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.9
                    successes = 0
                    success_count = 0

                # Coordinate-wise adaptation: increase sigma for dimensions that moved
                for d in range(dim):
                    if abs(noise[d]) > 1e-10:
                        # Scale diagonal based on improvement magnitude
                        diag[d] *= 1.05
                        diag[d] = min(diag[d], 5.0)
            else:
                # Failure
                success_count += 1

                if success_count >= window:
                    rate = successes / success_count
                    if rate < 0.1:
                        sigma *= 0.75
                    elif rate > 0.3:
                        sigma *= 1.1
                    successes = 0
                    success_count = 0

                # Coordinate-wise adaptation: decrease sigma for dimensions that didn't help
                for d in range(dim):
                    diag[d] *= 0.95
                    diag[d] = max(diag[d], 0.05)

            # Clamp sigma
            sigma = np.clip(sigma, 1e-4, 10.0)
            diag = np.clip(diag, 0.05, 5.0)

            # Occasionally try a larger exploratory step (restart-like behavior)
            if step % 50 == 0 and step < phase2_end:
                # Try a perturbation from best_x
                if time.time() < deadline - 10:
                    explore_sigma = 2.0 * phase_mult
                    y_explore = np.clip(best_x + explore_sigma * rng.standard_normal(dim), -5, 5)
                    fy_explore = func(y_explore)
                    calls += 1
                    if fy_explore < best_f:
                        best_f = fy_explore
                        best_x = y_explore.copy()
                        x = y_explore.copy()
                        fx = fy_explore
                        # Reset sigma to be larger after finding better point
                        sigma *= 1.5
                        diag = np.ones(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
