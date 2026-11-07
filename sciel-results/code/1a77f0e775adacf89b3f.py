"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+4)-ES with step-size adaptation and restart for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Best solution found across all restarts
        global_best_x = None
        global_best_f = np.inf

        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        # Multiple restarts to escape local minima
        # Rastrigin has many local minima, so restarts help
        n_restarts = 8
        budget_per_restart = max(1, self.budget // n_restarts)

        for restart in range(n_restarts):
            if calls_made >= self.budget:
                break
            if check_deadline():
                break

            # Initialize: first restart near origin, others random
            if restart == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)

            fx = func(x)
            calls_made += 1

            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

            # Step size for this restart
            sigma = 1.5

            # Population size
            pop_size = 4
            remaining = self.budget - calls_made

            if remaining <= 0:
                break

            # (1+4)-ES loop
            for _ in range(min(remaining, budget_per_restart)):
                if calls_made >= self.budget:
                    break
                if check_deadline():
                    break

                best_child_x = None
                best_child_f = np.inf

                for _ in range(pop_size):
                    if calls_made >= self.budget:
                        break
                    if check_deadline():
                        break

                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    calls_made += 1

                    if fy < best_child_f:
                        best_child_f = fy
                        best_child_x = y.copy()

                # Selection: take the best of parent and children
                if best_child_f < fx:
                    x = best_child_x
                    fx = best_child_f
                    # Step size adaptation: increase on success
                    sigma *= 1.25
                    if sigma > 5.0:
                        sigma = 5.0
                else:
                    # Step size adaptation: decrease on failure
                    sigma *= 0.8
                    if sigma < 0.01:
                        sigma = 0.01

                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()

        # Final refinement: local search around global best if we have budget left
        if global_best_x is not None and calls_made < self.budget and not check_deadline():
            x = global_best_x.copy()
            fx = global_best_f
            sigma = 0.1

            while calls_made < self.budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    if sigma > 0.5:
                        sigma = 0.5
                else:
                    sigma *= 0.9
                    if sigma < 0.001:
                        break

                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
