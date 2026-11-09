"""Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Simulated Annealing with adaptive cooling schedule, initialized from a broad uniform scout."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial broad uniform scout
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive temperature schedule
        # Start with a temperature that allows exploring the full range
        T = 50.0
        T_min = 1e-8
        cooling_rate = 0.995

        # Track steps for adaptive behavior
        accepted_uphill = 0
        total_steps = 0
        last_improvement_step = 0
        stall_counter = 0

        for i in range(budget - 1):
            total_steps += 1

            # Adaptive step size: starts large, shrinks over time but adapts to acceptance
            # Base step size decreases geometrically
            base_step = 2.0 * (0.995 ** i)
            # If we've been stuck, occasionally take a larger jump
            if stall_counter > 50:
                step_scale = 1.0 + 0.5 * np.exp(-stall_counter / 100.0)
            else:
                step_scale = 1.0

            step_size = base_step * step_scale

            # Generate candidate
            direction = rng.standard_normal(dim)
            y = np.clip(x + step_size * direction, -5, 5)
            fy = func(y)

            # Acceptance criterion
            if fy < fx:
                # Always accept downhill
                x = y
                fx = fy
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                    last_improvement_step = i
                    stall_counter = 0
                # Cool down a bit more aggressively when improving
                T *= cooling_rate
            else:
                # Uphill move: accept with probability exp(-(fy - fx) / T)
                delta_f = fy - fx
                if delta_f < T:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        x = y
                        fx = fy
                        accepted_uphill += 1
                        # Don't cool as much when accepting uphill
                        T *= 0.998
                    else:
                        stall_counter += 1
                        T *= cooling_rate
                else:
                    stall_counter += 1
                    T *= cooling_rate

            # Adaptive temperature floor and boost
            if T < T_min:
                T = T_min

            # If we've been stalled for a long time, boost temperature to escape
            if stall_counter > 100:
                T = max(T, 5.0)
                stall_counter = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
