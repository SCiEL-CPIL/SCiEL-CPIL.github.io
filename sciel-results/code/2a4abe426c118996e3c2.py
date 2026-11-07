"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-start and time guard for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        # Track calls
        call_count = 0

        def f(x):
            nonlocal call_count
            call_count += 1
            return func(np.asarray(x, dtype=float))

        def nelder_mead(func_eval, x0, max_calls, deadline_check):
            """Nelder-Mead simplex method."""
            n = len(x0)
            # Initial simplex
            simplex = [np.array(x0, dtype=float)]
            for i in range(n):
                dx = np.zeros(n)
                dx[i] = 0.25
                simplex.append(np.array(x0 + dx, dtype=float))
            fvals = [func_eval(s) for s in simplex]
            calls_used = n + 1

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_shrink = 0.5  # shrink

            while calls_used < max_calls:
                if deadline_check():
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func_eval(x_r)
                calls_used += 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func_eval(x_e)
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func_eval(x_c)
                    calls_used += 1
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fvals[i] = func_eval(simplex[i])
                            calls_used += 1
                            if calls_used >= max_calls:
                                break

            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx], calls_used

        # Multi-start strategy
        rng = np.random.default_rng(123)

        # Generate diverse starting points
        n_starts = 8
        starts = []
        # One at origin
        starts.append(np.zeros(d))
        # Random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, d))

        best_x = None
        best_f = np.inf
        calls_remaining = budget

        for start in starts:
            if calls_remaining <= 10:
                break
            if time.time() > deadline:
                break

            # Allocate calls to this start
            alloc = min(calls_remaining, max(50, budget // n_starts))

            x_opt, f_opt, used = nelder_mead(f, start, alloc, lambda: time.time() > deadline)

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt
            calls_remaining -= used

            if calls_remaining <= 0:
                break

        # If we have budget left, do a final local search around the best
        if best_x is not None and calls_remaining > 5 and time.time() < deadline:
            x_opt, f_opt, used = nelder_mead(f, best_x, calls_remaining, lambda: time.time() > deadline)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt

        return best_x, best_f
# EVOLVE-BLOCK-END
