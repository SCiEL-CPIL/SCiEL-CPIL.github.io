"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and CMA-ES-like covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

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

        # Phase 1: Coarse random scan to find promising basins
        n_init = min(20, total_budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, n)
            fx = safe_func(x)
            if time.time() > deadline - 20:
                break

        # Phase 2: (1+1)-ES with adaptive sigma and perturbation-based exploration
        x = best_x.copy()
        fx = best_f
        sigma = 1.5  # Start with a larger sigma for exploration

        window_size = 12
        window = []
        stagnation_count = 0
        max_stagnation = 25
        perturbation_count = 0
        max_perturbations = 8

        # Track best sigma for adaptation
        sigma_history = [sigma]

        while calls_made < total_budget and time.time() < deadline - 15:
            # Generate offspring
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = safe_func(y)

            # Adapt step size based on success
            if fy < fx:
                x, fx = y, fy
                window.append(1)
                stagnation_count = 0
            else:
                window.append(0)
                stagnation_count += 1

            if len(window) >= window_size:
                success_rate = sum(window) / window_size
                if success_rate > 0.35:
                    sigma *= 1.15
                elif success_rate < 0.08:
                    sigma *= 0.85
                window = window[1:]

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)

            # Perturbation-based exploration when stagnating
            if stagnation_count >= max_stagnation:
                perturbation_count += 1
                if perturbation_count <= max_perturbations:
                    # Jump to a random point but keep the best solution
                    # Use a moderate jump to explore new basins
                    jump_size = 2.0 + 0.5 * perturbation_count
                    jump = rng.uniform(-jump_size, jump_size, n)
                    new_x = np.clip(best_x + jump, -5, 5)
                    new_fy = safe_func(new_x)
                    
                    if new_fy < fx:
                        x, fx = new_x, new_fy
                        sigma = max(sigma, 1.0)
                    else:
                        # Stay at current x but increase sigma to escape
                        sigma = max(sigma, 1.5)
                    
                    stagnation_count = 0
                    window = []
                else:
                    # No more perturbations allowed, break out
                    break

        # Phase 3: Local refinement with smaller steps around the best
        if time.time() < deadline - 10 and calls_made < total_budget:
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
