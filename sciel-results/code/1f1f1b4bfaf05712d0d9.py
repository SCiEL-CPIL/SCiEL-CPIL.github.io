"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic gradient-free Powell's method with adaptive restarts
    to exploit local convexity quickly on BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return func(x)

        def powell_method(start_x, directions, remaining_budget):
            """Run Powell's method for a given number of budget."""
            nonlocal best_x, best_f, calls_made

            x = start_x.copy()
            f_x = wrapped_func(x)
            if calls_made >= budget:
                return x, f_x

            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()

            # directions is a list of dim vectors (initially unit basis)
            # We'll iterate: for each direction, do a line search
            # Then recompute directions based on the most improved direction

            while calls_made < budget:
                # For each direction, perform a 1D minimization
                old_x = x.copy()
                old_f = f_x
                min_f = np.inf
                min_idx = 0

                for i in range(dim):
                    if calls_made >= budget:
                        break

                    # Find minimum along direction[i]
                    # Use a ternary search or golden section
                    x_new, f_new = line_search_1d(x, directions[i], f_x, wrapped_func, budget)
                    x = x_new
                    f_x = f_new

                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()

                    if f_x < min_f:
                        min_f = f_x
                        min_idx = i

                if calls_made >= budget:
                    break

                # Check convergence
                improvement = old_f - f_x
                if abs(improvement) < 1e-12:
                    break

                # Update directions: replace the direction with the smallest improvement
                # with the direction from old_x to x
                delta_x = x - old_x
                if np.linalg.norm(delta_x) > 1e-12:
                    # Replace the direction that gave the smallest improvement
                    # Compute improvement for each direction
                    improvements = []
                    for i in range(dim):
                        # Estimate improvement from this direction
                        # We need to track per-direction improvements
                        pass

                    # Simpler approach: replace direction[min_idx] with normalized delta_x
                    # Actually in Powell's, we replace the direction that had the largest
                    # "cost" (smallest improvement). Let's track improvements.
                    # For simplicity, let's just replace the last direction with delta_x
                    # and shift directions
                    directions = directions[1:] + [delta_x / (np.linalg.norm(delta_x) + 1e-15)]

                # Re-center: do another round of line searches with updated directions
                # This is the "reinitialization" step in Powell's method
                # But we'll just continue with the updated directions

        def line_search_1d(x, direction, f_current, wrapped_func, budget):
            """Perform a 1D line search along direction starting from x."""
            nonlocal calls_made

            # Find the interval [t_low, t_high] that brackets the minimum
            # Start with a small step
            t = 0.1
            x_test = x + t * direction
            f_test = wrapped_func(x_test)

            if f_test >= f_current:
                # Try opposite direction
                t = -0.1
                x_test = x + t * direction
                f_test = wrapped_func(x_test)
                if f_test >= f_current:
                    return x, f_current

            # Now we know the minimum is between t=0 and t (or -t)
            t_low, t_high = 0.0, t if t > 0 else t
            f_low, f_high = f_current, f_test

            # If t < 0, swap
            if t < 0:
                t_low, t_high = t, 0.0
                f_low, f_high = f_test, f_current

            # Expand the interval
            for _ in range(5):
                if calls_made >= budget:
                    break
                t_new = t_high + (t_high - t_low)
                x_test = x + t_new * direction
                f_test = wrapped_func(x_test)
                if f_test < f_high:
                    t_high = t_new
                    f_high = f_test
                else:
                    break

            # Golden section search
            gr = (np.sqrt(5) - 1) / 2  # ~0.618
            t1 = t_low + (1 - gr) * (t_high - t_low)
            t2 = t_low + gr * (t_high - t_low)

            x1 = x + t1 * direction
            x2 = x + t2 * direction
            f1 = wrapped_func(x1)
            f2 = wrapped_func(x2)

            for _ in range(20):
                if calls_made >= budget:
                    break
                if f1 < f2:
                    t_high = t2
                    t2 = t1
                    f2 = f1
                    t1 = t_low + (1 - gr) * (t_high - t_low)
                    x1 = x + t1 * direction
                    f1 = wrapped_func(x1)
                else:
                    t_low = t1
                    t1 = t2
                    f1 = f2
                    t2 = t_low + gr * (t_high - t_low)
                    x2 = x + t2 * direction
                    f2 = wrapped_func(x2)

                if abs(t_high - t_low) < 1e-8:
                    break

            # Return the best point found
            t_best = (t_low + t_high) / 2
            x_best = x + t_best * direction
            f_best = wrapped_func(x_best)

            return x_best, f_best

        # Initial directions: unit basis vectors
        initial_directions = [np.eye(dim)[i] for i in range(dim)]

        # Multi-start with different starting points
        starts = [
            np.zeros(dim),
            np.array([1.0, -1.0, 2.0, -2.0, 1.5]),
            np.array([-2.0, 1.0, -1.0, 3.0, -1.5]),
            np.array([0.5, 0.5, -0.5, 0.5, -0.5]),
        ]

        # Allocate budget across starts
        n_starts = len(starts)
        budget_per_start = budget // n_starts

        for start_idx, start_x in enumerate(starts):
            if calls_made >= budget:
                break

            remaining = budget - calls_made
            if remaining < 10:
                break

            # Create fresh directions for each start
            directions = [np.eye(dim)[i] for i in range(dim)]

            powell_method(start_x, directions, remaining)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
