"""Combined optimizer: (1+1)-ES with adaptive restarts and multi-scale polish."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with hybrid sigma adaptation, adaptive restarts, and multi-scale polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Phase 0: Multi-point initialization for early exploration
        n_init = 5
        candidates = []
        for i in range(n_init):
            x_init = rng.uniform(-5, 5, n)
            fx_init = func(x_init)
            candidates.append((x_init, fx_init))

        best_idx = np.argmin([c[1] for c in candidates])
        x_best = candidates[best_idx][0].copy()
        fx_best = candidates[best_idx][1]
        calls_made = n_init
        remaining = budget - n_init

        # Track global best separately
        x_nm_best = x_best.copy()
        fx_nm_best = fx_best

        # ES parameters
        sigma = 2.0
        sigma_min = 1e-7
        sigma_max = 10.0
        window_size = 10
        successes_in_window = []
        stall_count = 0
        stall_limit = 30  # Adaptive restart threshold

        # Phase 1: Main (1+1)-ES with hybrid sigma adaptation and adaptive restarts
        while remaining > 0 and time.time() < deadline:
            # Generate candidate
            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1

            if fy < fx_best:
                # Success: accept and update
                x_best, fx_best = y, fy
                if fx_best < fx_nm_best:
                    x_nm_best, fx_nm_best = x_best.copy(), fx_best
                successes_in_window.append(1)
                stall_count = 0
                # Immediate sigma growth on success (1/5 rule)
                sigma = min(sigma * 1.15, sigma_max)
            else:
                # Failure: reject
                successes_in_window.append(0)
                stall_count += 1
                # Immediate sigma shrink on failure
                sigma = max(sigma * 0.9, sigma_min)

            # Update sliding window
            if len(successes_in_window) > window_size:
                successes_in_window.pop(0)

            # Window-based sigma correction
            if len(successes_in_window) == window_size:
                success_rate = np.mean(successes_in_window)
                # Target ~20% success rate (1/5 rule)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.05, sigma_max)
                elif success_rate < 0.15:
                    sigma = max(sigma * 0.95, sigma_min)

            # Adaptive restart on stagnation
            if stall_count >= stall_limit:
                stall_count = 0
                if remaining > 1:
                    # Restart: mix of local perturbation and global exploration
                    r = rng.random()
                    if r < 0.5:
                        # Local restart: perturb near current best
                        x_new = np.clip(x_best + 1.5 * sigma * rng.standard_normal(n), -5, 5)
                    else:
                        # Global restart: random point in box
                        x_new = rng.uniform(-5, 5, n)
                    fx_new = func(x_new)
                    calls_made += 1
                    remaining -= 1
                    if fx_new < fx_best:
                        x_best, fx_best = x_new, fx_new
                        if fx_best < fx_nm_best:
                            x_nm_best, fx_nm_best = x_best.copy(), fx_best
                    # Reset sigma to moderate value
                    sigma = 1.5
                    successes_in_window = []

        # Phase 2: Multi-scale polish around the best point
        if remaining > 0 and time.time() < deadline:
            x_polish = x_nm_best.copy()
            fx_polish = fx_nm_best

            # Multi-scale refinement with decreasing sigma
            scales = [0.1, 0.05, 0.01, 0.002, 0.0005, 0.0001]
            per_scale = max(remaining // len(scales), 1)

            for scale_idx, base_sigma in enumerate(scales):
                if remaining <= 0 or time.time() >= deadline:
                    break

                scale_sigma = base_sigma
                for i in range(min(per_scale, remaining)):
                    if time.time() >= deadline:
                        break

                    y = np.clip(x_polish + scale_sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    remaining -= 1

                    if fy < fx_polish:
                        x_polish, fx_polish = y, fy
                        scale_sigma *= 0.95  # Shrink faster on success
                        if fx_polish < fx_nm_best:
                            x_nm_best, fx_nm_best = x_polish.copy(), fx_polish
                    else:
                        scale_sigma *= 0.99

            # Final coordinate-wise line search for precision
            if remaining > 0 and time.time() < deadline:
                coord_steps = min(remaining, 40)
                step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]

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
