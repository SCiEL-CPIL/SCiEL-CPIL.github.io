"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-like adaptive step, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # We'll use a multi-restart (1+1)-ES with adaptive step size
        # and a simple population-based refinement phase.

        # Phase 1: Multi-restart (1+1)-ES to find a good region
        # Phase 2: Local refinement with smaller steps

        total_calls = 0

        def check_deadline():
            return time.time() > deadline

        # Use a set of restarts with different initial points
        # Budget allocation: give more budget to later restarts if early ones fail
        n_restarts = 5
        # Allocate budget: first restart gets less, later ones get more
        # Actually, let's just do a single long run with adaptive restarts

        # Strategy: Run (1+1)-ES with periodic restarts if no improvement
        # for a while. Track best so far.

        for restart in range(n_restarts):
            if check_deadline():
                break

            # Initialize
            if restart == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb from best found so far
                x = best_x + rng.normal(0, 1.0, dim)
                x = np.clip(x, -5, 5)

            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            no_improve = 0
            max_no_improve = 50 + restart * 10  # more patience for later restarts

            while total_calls < budget:
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 10.0)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

        # Phase 2: Local refinement around the best point
        if not check_deadline() and total_calls < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            no_improve = 0
            max_no_improve = 30

            while total_calls < budget:
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 2.0)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

        # Phase 3: Very fine local search
        if not check_deadline() and total_calls < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            max_no_improve = 20

            while total_calls < budget:
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.05, 0.5)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
