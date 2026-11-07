"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive stochastic trajectory optimizer with rank-one covariance updates
    for Lunacek bi-Rastrigin (f24). Uses a single continuous trajectory with
    adaptive step sizes and direction learning.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        # Phase 1: Random exploration to find promising basins
        rng = np.random.default_rng(42)
        n_random = min(50, budget // 10)
        candidates = []
        for i in range(n_random):
            if calls_made[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = counted_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Adaptive trajectory optimization from top starts
        # Use a rank-one adaptive method: maintain position, velocity, and adaptive step
        n_starts = min(3, len(candidates))
        for start_idx in range(n_starts):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            x = candidates[start_idx][1].copy()
            fx = candidates[start_idx][0]

            # Adaptive parameters
            step_size = 1.0  # Initial step size
            direction = np.zeros(dim)  # Accumulated direction
            momentum = 0.7  # Momentum coefficient
            lr = 0.1  # Learning rate for adaptive step
            min_step = 0.01
            max_step = 3.0

            # Track recent improvements
            recent_improvements = []
            stagnation_count = 0
            max_stagnation = 30

            while calls_made[0] < budget and time.time() < deadline:
                # Generate random perturbation direction
                noise = rng.standard_normal(dim)

                # Blend accumulated direction with new noise
                if np.linalg.norm(direction) < 1e-10:
                    direction = noise
                else:
                    direction = 0.8 * direction + 0.2 * noise
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 0:
                        direction /= norm_dir

                # Try step
                x_trial = x + step_size * direction
                x_trial = np.clip(x_trial, -5, 5)
                fx_trial = counted_func(x_trial)

                if fx_trial < fx:
                    # Success: update position and adapt
                    improvement = fx - fx_trial
                    recent_improvements.append(improvement)
                    if len(recent_improvements) > 20:
                        recent_improvements.pop(0)

                    # Update momentum direction
                    direction = 0.9 * direction + 0.1 * noise
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 0:
                        direction /= norm_dir

                    # Increase step size if improving
                    avg_improvement = np.mean(recent_improvements)
                    if avg_improvement > 0.01:
                        step_size = min(max_step, step_size * 1.1)
                    else:
                        step_size = min(max_step, step_size * 1.02)

                    x = x_trial
                    fx = fx_trial
                    stagnation_count = 0

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: decrease step size and adjust direction
                    stagnation_count += 1
                    step_size = max(min_step, step_size * 0.85)

                    # Rotate direction more aggressively
                    direction = 0.5 * direction + 0.5 * noise
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 0:
                        direction /= norm_dir

                    # If stuck too long, restart from current best
                    if stagnation_count >= max_stagnation:
                        break

            # Phase 3: Fine-tuning with smaller steps
            if calls_made[0] < budget and time.time() < deadline:
                fine_step = 0.1
                for i in range(50):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    for d in range(dim):
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        # Try positive and negative steps
                        for sign in [1, -1]:
                            if calls_made[0] >= budget or time.time() > deadline:
                                break
                            x_trial = x.copy()
                            x_trial[d] += sign * fine_step
                            x_trial = np.clip(x_trial, -5, 5)
                            fx_trial = counted_func(x_trial)
                            if fx_trial < fx:
                                x = x_trial
                                fx = fx_trial
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break
                        # Decrease step size gradually
                        fine_step *= 0.9

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counted_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
