"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule + adaptive restart + coordinate polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Multi-start (1+1)-ES with adaptive restart
        best_x = None
        best_f = np.inf
        total_calls = 0

        # We'll do multiple restarts, each with a sub-budget
        num_restarts = 4
        calls_per_restart = max(1, (budget - 50) // num_restarts)  # reserve 50 for polish

        for restart in range(num_restarts):
            if not time_ok():
                break

            # Initialize
            if restart == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb around best found so far
                x = best_x + rng.normal(0, 1.0, dim)
                x = np.clip(x, -5, 5)

            fx = func(x)
            total_calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            stall_count = 0
            stall_threshold = 20

            for i in range(calls_per_restart):
                if not time_ok():
                    break
                if total_calls >= budget - 50:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    stall_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    stall_count += 1

                    # Adaptive restart within a phase
                    if stall_count >= stall_threshold:
                        if sigma < 0.01:
                            break
                        # Try a larger step
                        sigma = max(sigma * 3.0, 0.5)
                        stall_count = 0

            # If we stalled hard, do a local restart around best
            if stall_count >= stall_threshold and time_ok() and total_calls < budget - 50:
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: Coordinate descent polish
        if time_ok() and best_x is not None and total_calls < budget:
            x = best_x.copy()
            fx = best_f
            remaining = budget - total_calls

            # Coordinate descent with adaptive step
            step = 0.5
            no_improve = 0
            max_no_improve = 20

            while total_calls < budget and time_ok() and no_improve < max_no_improve:
                improved = False
                for d in range(dim):
                    if not time_ok() or total_calls >= budget:
                        break

                    # Try positive and negative steps
                    for direction in [1, -1]:
                        if not time_ok() or total_calls >= budget:
                            break

                        y = x.copy()
                        y[d] += direction * step
                        y[d] = np.clip(y[d], -5, 5)
                        fy = func(y)
                        total_calls += 1

                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            step *= 1.5  # increase step if improving
                        else:
                            step *= 0.5  # decrease step

                    if step < 1e-6:
                        break

                if not improved:
                    no_improve += 1
                    step = max(step * 0.5, 1e-7)
                else:
                    no_improve = 0
                    step = min(step * 1.2, 2.0)

        # Phase 3: Final fine-tuning around best
        if time_ok() and best_x is not None and total_calls < budget:
            x = best_x.copy()
            fx = best_f
            step = 1e-3

            for _ in range(min(50, budget - total_calls)):
                if not time_ok() or total_calls >= budget:
                    break

                for d in range(dim):
                    if not time_ok() or total_calls >= budget:
                        break

                    for direction in [1, -1]:
                        if not time_ok() or total_calls >= budget:
                            break

                        y = x.copy()
                        y[d] += direction * step
                        y[d] = np.clip(y[d], -5, 5)
                        fy = func(y)
                        total_calls += 1

                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            step *= 0.5

                if step < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
