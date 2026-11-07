"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Powell's method with periodic restarts for bent-cigar f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Evaluate function with clipping
        def eval_func(x):
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Powell's method implementation
        def powell(x0, directions, max_evals, eval_counter):
            """
            Powell's method.
            x0: starting point
            directions: list of direction vectors (initially unit vectors)
            max_evals: remaining evaluations allowed
            eval_counter: [count] mutable list tracking evaluations used
            Returns: (best_x, best_f, evals_used)
            """
            n = len(x0)
            x = x0.copy()
            f = eval_func(x)
            eval_counter[0] += 1
            dirs = [d.copy() for d in directions]

            # Main loop
            while eval_counter[0] < max_evals and (time.time() < deadline):
                old_f = f
                min_idx = 0
                min_delta = 0.0
                x_start = x.copy()

                for i in range(n):
                    # Line search along direction i
                    d = dirs[i]
                    a = 1.0  # initial step
                    x_trial = x + a * d
                    f_trial = eval_func(x_trial)
                    eval_counter[0] += 1
                    if eval_counter[0] >= max_evals or time.time() >= deadline:
                        break

                    if f_trial < f:
                        # Try to find better step via golden section or parabolic
                        # Simple approach: try increasing and decreasing
                        f_prev = f
                        x_prev = x.copy()
                        x = x_trial
                        f = f_trial
                        # Try to expand
                        while eval_counter[0] < max_evals and time.time() < deadline:
                            x_new = x + a * d
                            f_new = eval_func(x_new)
                            eval_counter[0] += 1
                            if f_new >= f:
                                break
                            x_prev = x
                            f_prev = f
                            x = x_new
                            f = f_new
                        # If we went too far, do a simple bisection
                        if f_prev < f:
                            # We overshot, go back
                            x = x_prev
                            f = f_prev
                    else:
                        # Try negative direction
                        x_trial = x - a * d
                        f_trial = eval_func(x_trial)
                        eval_counter[0] += 1
                        if eval_counter[0] >= max_evals or time.time() >= deadline:
                            break
                        if f_trial < f:
                            f_prev = f
                            x_prev = x.copy()
                            x = x_trial
                            f = f_trial
                            while eval_counter[0] < max_evals and time.time() < deadline:
                                x_new = x - a * d
                                f_new = eval_func(x_new)
                                eval_counter[0] += 1
                                if f_new >= f:
                                    break
                                x_prev = x
                                f_prev = f
                                x = x_new
                                f = f_new
                            if f_prev < f:
                                x = x_prev
                                f = f_prev

                    delta = old_f - f
                    if delta > min_delta:
                        min_delta = delta
                        min_idx = i

                    if eval_counter[0] >= max_evals or time.time() >= deadline:
                        break

                if eval_counter[0] >= max_evals or time.time() >= deadline:
                    break

                # Update directions: replace the direction of the smallest improvement
                # with the direction from x_start to x
                new_dir = x - x_start
                norm = np.linalg.norm(new_dir)
                if norm > 1e-15:
                    dirs[min_idx] = new_dir / norm  # normalize
                else:
                    break

                # Check for convergence
                if abs(old_f - f) < 1e-12:
                    break

                # Reflect: do a line search along the new direction
                if min_delta > 0:
                    d_new = dirs[min_idx]
                    x_trial = x + min_delta * d_new
                    f_trial = eval_func(x_trial)
                    eval_counter[0] += 1
                    if eval_counter[0] < max_evals and f_trial < f:
                        # Expand
                        while eval_counter[0] < max_evals and time.time() < deadline:
                            x_new = x + min_delta * d_new
                            f_new = eval_func(x_new)
                            eval_counter[0] += 1
                            if f_new >= f:
                                break
                            x = x_new
                            f = f_new
                    elif eval_counter[0] < max_evals:
                        # Try negative
                        x_trial = x - min_delta * d_new
                        f_trial = eval_func(x_trial)
                        eval_counter[0] += 1
                        if f_trial < f:
                            while eval_counter[0] < max_evals and time.time() < deadline:
                                x_new = x - min_delta * d_new
                                f_new = eval_func(x_new)
                                eval_counter[0] += 1
                                if f_new >= f:
                                    break
                                x = x_new
                                f = f_new

            return x, f, eval_counter[0]

        # Initial unit directions
        unit_dirs = [np.eye(dim)[i] for i in range(dim)]

        # Strategy: multiple restarts with different starting points
        # Use a few good starting points, run Powell's on each
        rng = np.random.default_rng(42)

        # First, do a quick random search to find a good starting point
        best_x = None
        best_f = np.inf
        eval_counter = [0]
        max_initial = min(50, budget)

        for i in range(max_initial):
            if eval_counter[0] >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            eval_counter[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)
            eval_counter[0] += 1

        # Now run Powell's from the best point found
        x, f, used = powell(best_x, unit_dirs, budget, eval_counter)
        if f < best_f:
            best_f = f
            best_x = x

        # If we have budget left, try a few more restarts from different points
        remaining = budget - eval_counter[0]
        if remaining > 20 and time.time() < deadline:
            # Try starting from the origin and a few other points
            test_points = [
                np.zeros(dim),
                np.ones(dim) * 2.0,
                np.ones(dim) * -2.0,
                rng.uniform(-4, 4, dim),
            ]
            for tp in test_points:
                if eval_counter[0] >= budget or time.time() >= deadline:
                    break
                # Evaluate the test point
                f_tp = eval_func(tp)
                eval_counter[0] += 1
                if f_tp >= best_f:
                    continue
                # Run Powell's from this point
                x2, f2, used2 = powell(tp, unit_dirs, budget, eval_counter)
                if f2 < best_f:
                    best_f = f2
                    best_x = x2

        return best_x, best_f
# EVOLVE-BLOCK-END
