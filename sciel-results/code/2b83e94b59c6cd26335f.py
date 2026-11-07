"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with geometric cooling for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(789)

        def time_exceeded():
            return time.time() > deadline

        # Initialize at center
        x = np.zeros(dim)
        best_x = x.copy()
        best_f = func(x)
        evals = 1

        # Simulated Annealing parameters
        T_init = 50.0  # Large initial temperature for global exploration
        T_min = 1e-6
        alpha = 0.995  # Geometric cooling rate
        T = T_init

        # Step size schedule: start large, shrink over time
        step_size = 3.0  # Initial step size (covers most of [-5,5])
        step_decay = 0.999

        # Track stagnation for potential restarts
        stagnation_count = 0
        stagnation_threshold = 50

        while evals < budget and not time_exceeded():
            # Generate candidate
            y = rng.standard_normal(dim)
            x_new = x + step_size * y
            np.clip(x_new, -5, 5, out=x_new)

            f_new = func(x_new)
            evals += 1

            delta_f = f_new - best_f

            # Accept if better, or with probability based on temperature
            if delta_f < 0:
                x = x_new.copy()
                best_x = x_new.copy()
                best_f = f_new
                stagnation_count = 0
            else:
                # Probabilistic acceptance of worse solutions
                if T > 0:
                    accept_prob = np.exp(-delta_f / T)
                    if rng.random() < accept_prob:
                        x = x_new.copy()

            # Cool temperature
            T *= alpha
            if T < T_min:
                T = T_min

            # Decay step size
            step_size *= step_decay
            if step_size < 0.01:
                step_size = 0.01

            # If stuck for too long, try a random restart to escape local minima
            if not (delta_f < 0):
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count > stagnation_threshold and evals < budget - 50:
                # Random restart with moderate perturbation
                x = best_x + rng.standard_normal(dim) * 1.0
                np.clip(x, -5, 5, out=x)
                stagnation_count = 0
                # Slightly boost temperature to encourage exploration
                T = max(T, 1.0)
                step_size = max(step_size, 1.0)

        # Final polish: coordinate descent on best_x
        if evals < budget and not time_exceeded():
            remaining = budget - evals
            for _ in range(min(remaining, 200)):
                if evals >= budget or time_exceeded():
                    break
                for d in range(dim):
                    if evals >= budget or time_exceeded():
                        break
                    for delta in [0.005, -0.005, 0.001, -0.001]:
                        if evals >= budget or time_exceeded():
                            break
                        x_try = best_x.copy()
                        x_try[d] += delta
                        np.clip(x_try, -5, 5, out=x_try)
                        f_try = func(x_try)
                        evals += 1
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
