"""Optimized search for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES variant with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Track best overall
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Broad exploration with multiple starting points
        # Use a set of diverse starting points including origin and random
        n_starts = 6
        starts = []
        # Deterministic origin
        starts.append(np.zeros(d))
        # Random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, d))

        # Allocate budget per restart
        # Give more budget to earlier restarts, less to later ones
        # First restart gets ~50%, subsequent get decreasing amounts
        remaining = budget
        budgets = []
        for i in range(n_starts):
            if i == 0:
                b = int(remaining * 0.4)
            elif i == 1:
                b = int(remaining * 0.3)
            else:
                b = int(remaining / (n_starts - i))
            b = max(b, 20)
            b = min(b, remaining)
            budgets.append(b)
            remaining -= b

        for si, start in enumerate(starts):
            if not check_time() or calls_made >= budget:
                break

            x = start.copy()
            fx = evaluate(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES parameters
            sigma = 1.5
            C = np.eye(d)
            C_inv_sqrt = np.eye(d)  # approximate
            step_size = 1.0
            success_count = 0
            total_steps = 0
            sigma_min = 0.01
            sigma_max = 5.0

            # Use a simple evolution strategy with adaptive step size
            # and occasional random restarts within the phase
            local_best_x = x.copy()
            local_best_f = fx
            local_calls = 1
            max_local = budgets[si]

            while local_calls < max_local and check_time() and calls_made < budget:
                # Generate perturbation
                # Use a mix of Gaussian and coordinate-wise steps
                if rng.random() < 0.1:
                    # Coordinate-wise step for fine-tuning
                    axis = rng.integers(0, d)
                    step = rng.standard_normal(d)
                    step *= 0
                    step[axis] = sigma * rng.standard_normal()
                    y = np.clip(x + step, -5, 5)
                else:
                    y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)

                fy = evaluate(y)
                local_calls += 1

                if fy < local_best_f:
                    local_best_f = fy
                    local_best_x = y.copy()
                    x = y.copy()
                    fx = fy
                    # Increase step size on success
                    sigma = min(sigma * 1.2, sigma_max)
                    success_count += 1
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * 0.85, sigma_min)

                total_steps += 1
                # 1/5 success rule check every 10 steps
                if total_steps % 10 == 0:
                    if success_count / 10.0 > 0.2:
                        sigma = min(sigma * 1.5, sigma_max)
                    else:
                        sigma = max(sigma * 0.75, sigma_min)
                    success_count = 0

                # Early stopping if stuck
                if total_steps > 50 and sigma < 0.05:
                    break

            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # Phase 2: Fine-tuning around the best point
        if check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            fine_tune_budget = min(budget - calls_made, 100)

            for _ in range(fine_tune_budget):
                if not check_time() or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma = min(sigma * 1.1, 0.5)
                else:
                    sigma = max(sigma * 0.8, 0.005)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
