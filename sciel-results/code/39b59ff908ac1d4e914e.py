"""Portfolio search for BBOB f7 (Step ellipsoidal) with plateau-aware (1+1)-ES and random restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Optimizer:
    """Portfolio of (1+1)-ES with proper 1/5 success rule and plateau-aware restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Track global best
        best_x = rng.uniform(-4.0, 4.0, dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Random sampling to get a decent initial point (10% of budget)
        n_random = max(10, budget // 10)
        for i in range(n_random):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with proper 1/5 success rule and restarts
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # 1/5 success rule: maintain a window of last 15 outcomes
        window_size = 15
        outcome_window = deque(maxlen=window_size)
        no_improve = 0
        max_no_improve = 40
        restart_count = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate trial
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1

            # Record outcome in window
            if fy < fx - 1e-10:
                outcome_window.append(1)
                # Improvement
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
            else:
                outcome_window.append(0)
                no_improve += 1

            # Apply 1/5 success rule: adapt sigma based on mean of window
            if len(outcome_window) >= window_size:
                success_rate = sum(outcome_window) / len(outcome_window)
                if success_rate > 0.2:
                    sigma *= 1.1  # increase step size
                elif success_rate < 0.2:
                    sigma *= 0.9  # decrease step size
                # else: keep sigma unchanged (ideal case near 1/5)
            
            # Clamp sigma
            sigma = min(max(sigma, 0.01), 5.0)

            # Restart if stuck on plateau for too long
            if no_improve >= max_no_improve:
                restart_count += 1
                
                # Adaptive restart strategy:
                # Early restarts: more exploration (pure random)
                # Later restarts: more exploitation (perturb best)
                explore_prob = max(0.2, 0.7 - 0.1 * restart_count)
                
                if rng.random() < explore_prob:
                    x = rng.uniform(-5.0, 5.0, dim)
                else:
                    # Perturb global best with adaptive sigma
                    perturb_sigma = 1.0 + 0.5 * restart_count
                    x = best_x + perturb_sigma * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Reset ES state
                sigma = 1.0
                no_improve = 0
                outcome_window.clear()
                
                # Decrease patience for future restarts to encourage more exploration
                max_no_improve = max(20, 40 - 5 * restart_count)

        return best_x, best_f
# EVOLVE-BLOCK-END
