"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, periodic exploration, and polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Initial evaluation
        x_best = rng.uniform(-5, 5, n)
        fx_best = func(x_best)
        calls_made = 1
        remaining = budget - 1

        # Parameters
        sigma = 2.0
        sigma_min = 1e-5
        sigma_max = 8.0
        success_ratio = 0.2  # Target success rate for sigma adaptation
        stall_limit = 30
        stall_count = 0
        success_count = 0
        window_size = 10
        successes_in_window = []

        # Track best
        x_nm_best = x_best.copy()
        fx_nm_best = fx_best

        # Periodic exploration: every 40 calls, jump to a new random point
        explore_interval = 40
        calls_since_explore = 0

        # Phase 1: Main (1+1)-ES with periodic exploration
        while remaining > 0 and time.time() < deadline:
            # Periodic exploration: jump to new random point
            calls_since_explore += 1
            if calls_since_explore >= explore_interval:
                calls_since_explore = 0
                # Try a random jump
                x_jump = rng.uniform(-5, 5, n)
                fx_jump = func(x_jump)
                calls_made += 1
                remaining -= 1
                if fx_jump < fx_best:
                    x_best, fx_best = x_jump, fx_jump
                    if fx_best < fx_nm_best:
                        x_nm_best, fx_nm_best = x_best.copy(), fx_best
                    sigma = min(sigma * 1.5, sigma_max)
                    stall_count = 0
                else:
                    # Don't change sigma on failed exploration
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
                success_count += 1
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
                    sigma = min(sigma * 1.15, sigma_max)
                elif success_rate < success_ratio:
                    sigma = max(sigma * 0.85, sigma_min)

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

        # Phase 2: High-precision polish around the best point
        if remaining > 0 and time.time() < deadline:
            # Use remaining budget for fine-grained local search
            polish_budget = remaining
            polish_sigma = max(sigma * 0.3, 0.05)
            min_polish_sigma = 1e-5

            for i in range(polish_budget):
                if time.time() >= deadline:
                    break
                # Decrease step size gradually
                progress = i / max(polish_budget - 1, 1)
                current_sigma = polish_sigma * (1.0 - 0.95 * progress)
                current_sigma = max(current_sigma, min_polish_sigma)

                y = np.clip(x_nm_best + current_sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)

                if fy < fx_nm_best:
                    x_nm_best, fx_nm_best = y, fy

        return x_nm_best, fx_nm_best
# EVOLVE-BLOCK-END
