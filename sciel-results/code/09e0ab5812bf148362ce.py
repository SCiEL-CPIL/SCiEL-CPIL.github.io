"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size and time-based deadline guard.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Multi-start with diverse initial points
        # Use a set of starting points spread across the domain
        n_starts = 8
        starts = []
        # One at origin
        starts.append(np.zeros(self.dim))
        # Others at random locations
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))

        # Allocate budget across starts: give more to early starts, less to later
        # but ensure each gets a minimum number of evaluations
        # Total budget: 500
        # Let's do: first start gets 100, next 4 get 70 each (280), last 3 get ~40 each (120)
        # Total: 100 + 280 + 120 = 500
        allocations = [100, 70, 70, 70, 70, 40, 40, 40]
        # Adjust if sum != budget
        while sum(allocations) < self.budget:
            allocations[-1] += 1
        while sum(allocations) > self.budget:
            allocations[-1] -= 1

        call_count = 0

        for i, x0 in enumerate(starts):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            alloc = allocations[i]
            x = x0.copy()
            fx = func(x)
            call_count += 1
            update_best(x, fx)

            sigma = 1.5  # initial step size
            no_improve = 0
            max_no_improve = 30  # restart this sub-run if stuck

            for step in range(alloc - 1):
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Adaptive sigma: increase if improving, decrease if not
                # Use 1/5 success rule inspired logic
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    no_improve = 0
                    sigma *= 1.2  # increase step size on success
                else:
                    no_improve += 1
                    sigma *= 0.85  # decrease step size on failure

                # If stuck for too long, try a larger jump (restart within sub-run)
                if no_improve >= max_no_improve:
                    # Make a larger exploratory step
                    y = np.clip(x + sigma * 3 * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    call_count += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        update_best(x, fx)
                        no_improve = 0
                        sigma *= 1.0
                    else:
                        no_improve = 0
                        sigma *= 0.5

                # Keep sigma within reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
