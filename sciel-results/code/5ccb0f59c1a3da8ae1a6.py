"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with window-based success-rate adaptation, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 0: Multi-start initialization to find both basins of Lunacek
        # Lunacek bi-Rastrigin has a global minimum near 0 and a local minimum at offset
        init_points = [
            np.zeros(n),
            rng.uniform(-1, 1, n),
            rng.uniform(-2, 2, n),
            np.ones(n) * 2.5,
            np.ones(n) * -2.5,
            rng.uniform(-4, 4, n),
        ]
        for xp in init_points:
            if calls_made >= total_budget:
                break
            safe_func(xp)

        # Phase 1: Main (1+1)-ES with success-rate-based adaptation and restarts
        # Use a window-based success rate for step size adaptation
        max_stagnation_per_run = 35
        max_restarts = 6

        for restart in range(max_restarts):
            if calls_made >= total_budget or time.time() >= deadline - 10:
                break

            # Start from best point with perturbation for later restarts
            if restart == 0:
                x = best_x.copy()
            else:
                # Perturb best point to escape local minimum
                perturb_scale = 0.5 + 0.3 * restart
                x = best_x + perturb_scale * rng.standard_normal(n)
                x = np.clip(x, -5, 5)

            fx = safe_func(x)
            if fx >= best_f:
                # If perturbation didn't help, start from best
                x = best_x.copy()
                fx = best_f

            sigma = 1.0 if restart == 0 else 0.8 + 0.2 * restart
            stagnation_count = 0

            # Success rate based step size adaptation
            window_size = 15
            window = []

            while calls_made < total_budget and time.time() < deadline - 5:
                # Generate offspring
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                # Adapt step size based on success rate
                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                    stagnation_count = 0
                else:
                    window.append(0)
                    stagnation_count += 1

                if len(window) >= window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    window = window[1:]

                # Check for stagnation
                if stagnation_count >= max_stagnation_per_run:
                    break

        # Phase 2: Local refinement with aggressive step size decay
        if time.time() < deadline - 5 and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            stagnation_count = 0
            max_stagnation = 20
            window = []
            window_size = 10

            remaining = total_budget - calls_made
            for step in range(remaining):
                if time.time() > deadline - 5:
                    break

                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    stagnation_count = 0
                    window.append(1)
                else:
                    stagnation_count += 1
                    window.append(0)

                if len(window) >= window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.4:
                        sigma *= 1.05
                    elif success_rate < 0.1:
                        sigma *= 0.95
                    window = window[1:]

                if stagnation_count >= max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
