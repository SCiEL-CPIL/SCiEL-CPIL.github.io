"""Simulated Annealing optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses a geometric cooling schedule with large initial step sizes and occasional
reheat to escape local minima. This avoids the covariance adaptation overhead
of CMA-ES while leveraging probabilistic uphill acceptance to navigate the
ill-conditioned landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Simulated Annealing for ill-conditioned BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def check_deadline():
            return time.time() > deadline - 5.0 or total_calls >= budget

        # Phase 1: Initial random sampling to find a good starting point
        n_samples = min(20, budget // 10)
        samples = rng.uniform(-4, 4, size=(n_samples, dim))
        sample_values = []
        for i in range(n_samples):
            if check_deadline():
                break
            x = samples[i]
            fx = func(x)
            total_calls += 1
            sample_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if not sample_values:
            return best_x, best_f

        # Phase 2: Simulated Annealing
        # Start from the best initial sample
        current_x = best_x.copy()
        current_f = best_f

        # SA parameters
        T_init = 10.0
        T_min = 1e-8
        alpha = 0.995  # cooling rate
        max_steps = budget - total_calls

        # Adaptive step size
        step_scale = 1.5

        # Track consecutive non-improving steps for reheat
        no_improve_count = 0
        reheat_threshold = 50
        reheat_factor = 0.5

        T = T_init
        step = 0

        while step < max_steps and not check_deadline():
            # Generate a new candidate
            # Use a mix of large and small steps
            if rng.random() < 0.1:
                # Occasionally take a large jump to escape local minima
                delta = rng.uniform(-step_scale, step_scale, size=dim)
            else:
                # Normal Gaussian step
                delta = rng.standard_normal(dim) * step_scale * 0.5

            new_x = np.clip(current_x + delta, -5, 5)
            new_f = func(new_x)
            total_calls += 1
            step += 1

            # Accept or reject
            if new_f < current_f:
                # Accept improvement
                current_x = new_x
                current_f = new_f
                no_improve_count = 0

                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
            else:
                # Accept with probability exp(-delta_f / T)
                delta_f = new_f - current_f
                if delta_f < 0:
                    accept_prob = 1.0
                else:
                    if T > 1e-10:
                        accept_prob = np.exp(-delta_f / T)
                    else:
                        accept_prob = 0.0

                if rng.random() < accept_prob:
                    current_x = new_x
                    current_f = new_f
                    no_improve_count += 1
                else:
                    no_improve_count += 1

            # Cool down
            T *= alpha

            # Reheat if stuck
            if no_improve_count >= reheat_threshold and T < T_init * reheat_factor:
                T = T_init * reheat_factor
                no_improve_count = 0
                # Also perturb the current position
                current_x = np.clip(current_x + rng.standard_normal(dim) * 0.5, -5, 5)
                current_f = func(current_x)
                total_calls += 1
                step += 1
                if current_f < best_f:
                    best_f = current_f
                    best_x = current_x.copy()

            # Adapt step size
            if no_improve_count > 100:
                step_scale *= 0.8
            elif no_improve_count == 0:
                step_scale = min(step_scale * 1.05, 3.0)

            # Check deadline
            if check_deadline():
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
