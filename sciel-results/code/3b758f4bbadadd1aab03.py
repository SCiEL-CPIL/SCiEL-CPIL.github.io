"""Optimizer for BBOB f6 (Attractive Sector) using (1+1)-ES with adaptive step size."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES optimizer with adaptive step size and global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls_used = 0

        # Phase 1: Global exploration - sample diverse points
        best_x = np.zeros(dim)
        best_f = np.inf

        # Start with origin and a few structured points
        initial_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ]

        for x in initial_points:
            if time.time() > deadline or calls_used >= budget:
                break
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Add random points for global exploration
        n_random = 20
        for _ in range(n_random):
            if time.time() > deadline or calls_used >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive step size
        x = best_x.copy()
        f = best_f
        sigma = 1.0  # Initial step size
        success_count = 0
        total_steps = 0
        last_improvement = 0

        # Track history for step size adaptation
        step_history = []

        while calls_used < budget and time.time() < deadline:
            # Generate a single offspring
            z = rng.standard_normal(dim)
            x_off = x + sigma * z
            x_off = np.clip(x_off, -5.0, 5.0)

            f_off = float(func(x_off))
            calls_used += 1
            total_steps += 1

            # Record step outcome
            improved = f_off < f
            step_history.append(improved)

            if improved:
                x = x_off
                f = f_off
                success_count += 1
                last_improvement = total_steps
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # Adapt step size every 10 steps
            if total_steps % 10 == 0:
                # Count successes in last 10 steps
                recent = step_history[-10:]
                n_success = sum(recent)

                # Adaptive rule: increase sigma if too many successes, decrease if too few
                if n_success > 6:
                    sigma *= 1.5
                elif n_success < 3:
                    sigma *= 0.7
                else:
                    sigma *= 1.0

                # Keep sigma in reasonable bounds
                sigma = max(1e-6, min(sigma, 10.0))

            # Check for stagnation - if no improvement in many steps, restart with larger sigma
            if total_steps - last_improvement > 50 and sigma < 5.0:
                sigma = min(sigma * 2.0, 5.0)
                last_improvement = total_steps

            # If sigma is very small, we've likely converged
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
