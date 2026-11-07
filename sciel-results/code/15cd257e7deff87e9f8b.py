"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, smart restarts, and multi-phase search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f, best_x = f, x.copy()

        # Phase 0: Quick initial sampling to find a good starting basin
        # Evaluate origin and a few random points
        candidates = [np.zeros(d)]
        for _ in range(4):
            candidates.append(rng.uniform(-4.5, 4.5, d))

        for c in candidates:
            if check_deadline():
                break
            fc = func(c)
            total_calls += 1
            update_best(c, fc)

        # Start ES from the best initial point
        x = best_x.copy()
        fx = best_f

        # Phase 1: Main (1+1)-ES with adaptive sigma and smart restarts
        search_budget = budget - max(30, budget // 20)
        sigma = 2.0
        min_sigma = 1e-6
        max_sigma = 3.0
        no_improve = 0
        max_no_improve = 45
        restart_count = 0
        max_restarts = 6
        success_history = []
        success_window = 10

        while total_calls < search_budget and not check_deadline():
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                # Success
                x, fx = y, fy
                no_improve = 0
                success_history.append(1)
                update_best(x, fx)
            else:
                # Failure
                no_improve += 1
                success_history.append(0)

            # Adaptive sigma based on recent success rate
            if len(success_history) >= success_window:
                recent = success_history[-success_window:]
                success_rate = sum(recent) / len(recent)
                # Target success rate ~0.33 for (1+1)-ES
                if success_rate > 0.4:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.85
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, min_sigma, max_sigma)
                success_history = success_history[-(success_window - 1):]

            # Smart restart strategy
            if no_improve >= max_no_improve:
                if restart_count < max_restarts:
                    restart_count += 1
                    # Try a moderate perturbation first
                    perturb_sigma = max(sigma * 2, 1.0)
                    y_pert = np.clip(x + perturb_sigma * rng.standard_normal(d), -5, 5)
                    fy_pert = func(y_pert)
                    total_calls += 1

                    if fy_pert < fx:
                        x, fx = y_pert, fy_pert
                        update_best(x, fx)
                        sigma = max(sigma * 1.5, 1.0)
                    else:
                        # Full random restart
                        x = rng.uniform(-4.5, 4.5, d)
                        fx = func(x)
                        total_calls += 1
                        update_best(x, fx)
                        sigma = 1.5

                    no_improve = 0
                    success_history = []

        # Phase 2: Final polish around the best point
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            polish_budget = budget - total_calls
            polish_steps = min(max(30, budget // 20), polish_budget)

            for _ in range(polish_steps):
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.05
                else:
                    sigma *= 0.92

                if sigma < 1e-9:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
