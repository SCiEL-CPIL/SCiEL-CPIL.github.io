"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, periodic exploration, and multi-scale polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Multi-point initialization for better early exploration
        n_init = min(5, budget)
        candidates = []
        for i in range(n_init):
            x_init = rng.uniform(-5, 5, n)
            fx_init = func(x_init)
            candidates.append((x_init, fx_init))
        
        # Select best initial point
        best_idx = np.argmin([c[1] for c in candidates])
        x_best = candidates[best_idx][0].copy()
        fx_best = candidates[best_idx][1]
        calls_made = n_init
        remaining = budget - n_init

        # Parameters
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 8.0
        success_ratio = 0.2  # Target success rate for sigma adaptation
        stall_limit = 25
        stall_count = 0
        window_size = 10
        successes_in_window = []

        # Track best
        x_nm_best = x_best.copy()
        fx_nm_best = fx_best

        # Periodic exploration: every 35 calls, jump to a new random point
        explore_interval = 35
        calls_since_explore = 0

        # Phase 1: Main (1+1)-ES with periodic exploration
        while remaining > 0 and time.time() < deadline:
            # Periodic exploration: jump to a new random point
            calls_since_explore += 1
            if calls_since_explore >= explore_interval:
                calls_since_explore = 0
                # Mix of global and local exploration
                if rng.random() < 0.4:
                    # Global exploration
                    x_jump = rng.uniform(-5, 5, n)
                else:
                    # Local exploration with moderate step
                    x_jump = np.clip(x_best + 3.0 * rng.standard_normal(n), -5, 5)
                fx_jump = func(x_jump)
                calls_made += 1
                remaining -= 1
                if fx_jump < fx_best:
                    x_best, fx_best = x_jump, fx_jump
                    if fx_best < fx_nm_best:
                        x_nm_best, fx_nm_best = x_best.copy(), fx_best
                    sigma = min(sigma * 1.3, sigma_max)
                    stall_count = 0
                else:
                    pass
                continue

            # Generate candidate
            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1

            if fy < fx_best:
                x_best, fx_best = y, fy
                if fx_best < fx_nm_best:
                    x_nm_best, fx_nm_best = x_best.copy(), fx_best
                successes_in_window.append(1)
                stall_count = 0
            else:
                successes_in_window.append(0)

            # Update window
            if len(successes_in_window) > window_size:
                successes_in_window.pop(0)

            # Adapt sigma based on success rate in recent window
            if len(successes_in_window) >= window_size:
                success_rate = np.mean(successes_in_window)
                if success_rate > success_ratio:
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_rate < success_ratio:
                    sigma = max(sigma * 0.8, sigma_min)

            # Stall detection
            if fy >= fx_best:
                stall_count += 1
                if stall_count >= stall_limit:
                    # Restart from new random point
                    stall_count = 0
                    sigma = 2.0
                    x_new = rng.uniform(-5, 5, n)
                    fx_new = func(x_new)
                    calls_made += 1
                    remaining -= 1
                    if fx_new < fx_best:
                        x_best, fx_best = x_new, fx_new
                        if fx_best < fx_nm_best:
                            x_nm_best, fx_nm_best = x_best.copy(), fx_best
                    # Reset window
                    successes_in_window = []

        # Phase 2: Multi-scale polish around the best point
        if remaining > 0 and time.time() < deadline:
            polish_budget = remaining
            x_polish = x_nm_best.copy()
            fx_polish = fx_nm_best

            # Multi-scale refinement: gradually reduce sigma in stages
            scales = [max(sigma * 0.2, 0.1), 0.05, 0.01, 0.002, 0.0005, 0.0001, 0.00005]
            per_scale = max(polish_budget // len(scales), 1)

            for scale_idx, base_sigma in enumerate(scales):
                if remaining <= 0 or time.time() >= deadline:
                    break

                # For each scale, do a mini (1+1)-ES with fixed sigma
                scale_sigma = base_sigma
                for i in range(min(per_scale, remaining)):
                    if time.time() >= deadline:
                        break

                    y = np.clip(x_polish + scale_sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    remaining -= 1

                    if fy < fx_polish:
                        x_polish, fx_polish = y, fy
                        # Slowly decrease sigma within this scale
                        scale_sigma *= 0.98
                    else:
                        scale_sigma *= 0.995

                    if fx_polish < fx_nm_best:
                        x_nm_best, fx_nm_best = x_polish.copy(), fx_polish

            # Final coordinate-wise line search for precision
            if remaining > 0 and time.time() < deadline:
                coord_steps = min(remaining, 60)
                step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

                for cs in range(coord_steps):
                    if remaining <= 0 or time.time() >= deadline:
                        break
                    step = step_sizes[min(cs, len(step_sizes) - 1)]

                    for d in range(n):
                        if remaining <= 0 or time.time() >= deadline:
                            break
                        for direction in [1, -1]:
                            if remaining <= 0 or time.time() >= deadline:
                                break
                            y = x_polish.copy()
                            y[d] = np.clip(y[d] + direction * step, -5, 5)
                            fy = func(y)
                            remaining -= 1
                            if fy < fx_polish:
                                x_polish, fx_polish = y, fy
                                if fx_polish < fx_nm_best:
                                    x_nm_best, fx_nm_best = x_polish.copy(), fx_polish
                                break

        return x_nm_best, fx_nm_best
# EVOLVE-BLOCK-END
