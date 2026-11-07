"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Simulated Annealing with Local Refinement for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point - start at origin
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Simulated Annealing Parameters
        # f19 is Griewank-Rosenbrock. Range [-5, 5].
        # Initial temperature should be high enough to accept large jumps.
        # Typical function value range for f19 is [0, ~100].
        T = 30.0
        T_min = 0.05
        # Cooling rate: T = T * cooling_rate
        # We want to reach T_min in about 60% of the budget.
        sa_steps = int(budget * 0.6)
        cooling_rate = (T_min / T) ** (1.0 / max(1, sa_steps))

        # Adaptive step size - start moderate
        step_size = 1.5

        # Track acceptance rate for adaptive step sizing
        accept_count = 0
        total_count = 0
        window_size = 20

        # Phase 1: Simulated Annealing
        for i in range(sa_steps):
            if calls >= budget:
                break

            # Generate neighbor
            # Use Gaussian noise scaled by step_size
            noise = rng.standard_normal(n) * step_size
            x_new = x + noise
            x_new = np.clip(x_new, -5, 5)

            fx_new = func(x_new)
            calls += 1
            total_count += 1

            # Acceptance criterion
            if fx_new < fx:
                x = x_new
                fx = fx_new
                accept_count += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
            else:
                # Accept worse solution with probability exp(-delta / T)
                delta = fx_new - fx
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    x = x_new
                    fx = fx_new
                    accept_count += 1

            # Update temperature
            T *= cooling_rate

            # Adaptive step size based on acceptance rate
            if total_count >= window_size:
                accept_rate = accept_count / total_count
                # If acceptance rate is high, increase step size
                # If acceptance rate is low, decrease step size
                if accept_rate > 0.7:
                    step_size *= 1.1
                elif accept_rate < 0.3:
                    step_size *= 0.9
                # Reset counters
                accept_count = 0
                total_count = 0

            # Ensure step size stays in reasonable bounds
            step_size = np.clip(step_size, 0.05, 3.0)

        # Phase 2: Local Refinement with coordinate descent
        # Use remaining budget for local search around best_x
        local_budget = budget - calls
        if local_budget > 0:
            x_local = best_x.copy()
            fx_local = best_f
            # Start with moderate step for local search
            local_step = 0.5

            # Try coordinate axes first, then random directions
            for i in range(local_budget):
                if calls >= budget:
                    break

                # Alternate between coordinate axes and random directions
                if i % 3 == 0:
                    # Pick a random coordinate axis
                    axis = rng.integers(n)
                    direction = np.zeros(n)
                    direction[axis] = 1.0
                elif i % 3 == 1:
                    # Pick a random coordinate axis (negative)
                    axis = rng.integers(n)
                    direction = np.zeros(n)
                    direction[axis] = -1.0
                else:
                    # Random direction
                    direction = rng.standard_normal(n)
                    direction /= np.linalg.norm(direction)

                x_try = x_local + local_step * direction
                x_try = np.clip(x_try, -5, 5)

                fx_try = func(x_try)
                calls += 1

                if fx_try < fx_local:
                    x_local = x_try
                    fx_local = fx_try
                    if fx_try < best_f:
                        best_f = fx_try
                        best_x = x_local.copy()
                    # Keep step size or slightly increase if improving
                    local_step = min(local_step * 1.05, 2.0)
                else:
                    # Try opposite direction
                    x_try2 = x_local - local_step * direction
                    x_try2 = np.clip(x_try2, -5, 5)

                    if calls + 1 <= budget:
                        fx_try2 = func(x_try2)
                        calls += 1

                        if fx_try2 < fx_local:
                            x_local = x_try2
                            fx_local = fx_try2
                            if fx_try2 < best_f:
                                best_f = fx_try2
                                best_x = x_local.copy()
                            local_step = min(local_step * 1.05, 2.0)
                        else:
                            # Reduce step size
                            local_step *= 0.7

                    if local_step < 0.01:
                        # Reset step size periodically
                        if i % 30 == 0:
                            local_step = 0.3

        return best_x, best_f
# EVOLVE-BLOCK-END
