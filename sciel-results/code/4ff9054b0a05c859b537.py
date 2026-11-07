"""Optimizer for BBOB f22 (Gallagher 21 peaks) using Powell's method with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's method with multiple restarts for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        # Track best result
        best_x = None
        best_f = float('inf')
        eval_count = 0

        def eval_func(x):
            nonlocal eval_count
            x = np.clip(x, -5, 5)
            val = func(x)
            eval_count += 1
            return val

        # Powell's method implementation
        def powell(x0, budget_remaining):
            """
            Powell's method for minimization.
            Returns (best_x, best_f, evals_used)
            """
            n = self.dim
            x = np.array(x0, dtype=float)
            f = eval_func(x)
            evals = 1

            # Initial directions: coordinate axes
            directions = np.eye(n)

            # Powell's method loop
            while evals < budget_remaining:
                # Check time
                if time.time() > deadline:
                    break

                x_old = x.copy()
                f_old = f

                # Step 1: Minimize along each direction
                for i in range(n):
                    if evals >= budget_remaining:
                        break
                    if time.time() > deadline:
                        break

                    d = directions[i]

                    # Find minimum along direction d using golden section or parabolic interpolation
                    # First, determine the direction sign and initial step
                    # Try a small step to determine direction
                    step = 0.1
                    x_test = x + step * d
                    f_test = eval_func(x_test)
                    evals += 1

                    if f_test < f:
                        # Move in positive direction
                        d = d
                        x_prev = x
                        f_prev = f
                        x = x_test
                        f = f_test
                    else:
                        # Move in negative direction
                        d = -d
                        x_prev = x
                        f_prev = f
                        x = x - step * d  # = x + step*(-d)
                        f = eval_func(x)
                        evals += 1

                    # Now do parabolic/golden section search along d
                    # Use a simple approach: try expanding step if still improving
                    step = 0.1
                    while evals < budget_remaining and time.time() < deadline:
                        x_next = x + step * d
                        f_next = eval_func(x_next)
                        evals += 1

                        if f_next < f:
                            x_prev = x
                            f_prev = f
                            x = x_next
                            f = f_next
                            step *= 1.5
                        else:
                            break

                    # Parabolic interpolation for refinement
                    if evals < budget_remaining and time.time() < deadline:
                        # We have x_prev, x, and we know the step direction
                        # Use quadratic interpolation
                        # x_prev, x, and x - step*d (the one that failed)
                        x3 = x - step * d
                        f3 = eval_func(x3)
                        evals += 1

                        # Three points: x_prev, x, x3
                        # Actually, let's use a simpler approach:
                        # Do a few more line searches with smaller steps
                        for _ in range(3):
                            if evals >= budget_remaining:
                                break
                            if time.time() > deadline:
                                break
                            # Try halving the step
                            for trial_step in [step * 0.5, step * 0.25]:
                                x_try = x + trial_step * d
                                f_try = eval_func(x_try)
                                evals += 1
                                if f_try < f:
                                    x = x_try
                                    f = f_try
                                    step = trial_step
                                    break
                                x_try = x - trial_step * d
                                f_try = eval_func(x_try)
                                evals += 1
                                if f_try < f:
                                    x = x_try
                                    f = f_try
                                    step = trial_step
                                    break
                            else:
                                break

                # Check if we made progress
                if np.allclose(x, x_old, atol=1e-10):
                    break

                # Step 2: Check if we should reorder directions
                # The direction that gave the largest improvement should be moved to the end
                # For simplicity, we'll just keep the directions as is for now
                # A more sophisticated approach would track improvements per direction

            return x, f, evals

        # Multiple restarts with different initial points
        rng = np.random.default_rng(42)

        # Allocate budget across restarts
        # Start with a few restarts, then refine
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts

        for r in range(num_restarts):
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break

            # Generate initial point
            if r == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)

            # Run Powell's method
            x_best, f_best, evals_used = powell(x0, self.budget - eval_count)

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

            # If we found a good solution, do additional refinement
            if eval_count < self.budget and time.time() < deadline:
                # Do another round of Powell from the best point
                remaining = self.budget - eval_count
                if remaining > 10:
                    x_ref, f_ref, evals_ref = powell(best_x, remaining)
                    if f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()

        # Final check
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = eval_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
