"""Nelder-Mead simplex with adaptive reset and multi-start diversification for f24 (Lunacek bi-Rastrigin)."""
import numpy as np
import time


class Optimizer:
    """
    Derivative-free trust-region optimizer using Nelder-Mead simplex method
    with adaptive reset and multi-start diversification. Models the landscape
    as a polytope (simplex) rather than a Gaussian, better suited for the
    non-smooth Rastrigin component and skewed Lunacek basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        def nelder_mead(x0, max_evals, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """
            Nelder-Mead simplex optimization.
            Returns (best_x, best_f, evals_used)
            """
            n = len(x0)
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.1 * max(1.0, abs(x0[i]))

            # Evaluate initial simplex
            vals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls_made[0] >= budget or time.time() > deadline:
                    return simplex[0].copy(), vals[0], calls_made[0]
                vals[i] = counted_func(simplex[i])

            # Sort simplex
            order = np.argsort(vals)
            simplex = simplex[order]
            vals = vals[order]

            best_local_x = simplex[0].copy()
            best_local_f = vals[0]
            evals_used = n + 1

            # Nelder-Mead iterations
            max_iter = max_evals // (n + 1) + 10
            for it in range(max_iter):
                if calls_made[0] >= budget or time.time() > deadline:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                f_ref = counted_func(x_ref)
                evals_used += 1

                if f_ref < vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    f_exp = counted_func(x_exp)
                    evals_used += 1

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        vals[-1] = f_ref
                elif f_ref < vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < vals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    x_con = np.clip(x_con, -5, 5)
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    f_con = counted_func(x_con)
                    evals_used += 1

                    if f_con <= min(f_ref, vals[-1]):
                        simplex[-1] = x_con
                        vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls_made[0] >= budget or time.time() > deadline:
                                break
                            vals[i] = counted_func(simplex[i])
                            evals_used += 1
                        if calls_made[0] >= budget or time.time() > deadline:
                            break

                # Re-sort
                order = np.argsort(vals)
                simplex = simplex[order]
                vals = vals[order]

                # Track best
                if vals[0] < best_local_f:
                    best_local_f = vals[0]
                    best_local_x = simplex[0].copy()

                # Check convergence
                if np.std(vals) < 1e-10:
                    break

            return best_local_x, best_local_f, evals_used

        # Multi-start strategy
        # Start 1: Origin
        x0 = np.zeros(dim)
        if calls_made[0] < budget and time.time() < deadline:
            f0 = counted_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Start 2: Random point
        x0 = rng.uniform(-4, 4, dim)
        if calls_made[0] < budget and time.time() < deadline:
            f0 = counted_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Run Nelder-Mead from multiple starting points
        # Allocate budget for multiple restarts
        restarts = 5
        evals_per_restart = max(50, (budget - calls_made[0]) // restarts)

        for r in range(restarts):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            # Choose starting point
            if r == 0:
                # Start from current best
                x_start = best_x.copy() if best_x is not None else np.zeros(dim)
            elif r == 1:
                # Start from a perturbation of best
                x_start = best_x + rng.normal(0, 0.5, dim) if best_x is not None else rng.uniform(-3, 3, dim)
                x_start = np.clip(x_start, -5, 5)
            elif r == 2:
                # Start from another random point
                x_start = rng.uniform(-4, 4, dim)
            elif r == 3:
                # Start near origin with small perturbation
                x_start = rng.normal(0, 1.0, dim)
            else:
                # Start from a grid-like point
                x_start = np.array([rng.uniform(-4, 4) for _ in range(dim)])

            x_start = np.clip(x_start, -5, 5)

            # Run Nelder-Mead
            x_best_r, f_best_r, evals_used = nelder_mead(x_start, evals_per_restart)

            if f_best_r < best_f:
                best_f = f_best_r
                best_x = x_best_r.copy()

        # Final polish: local coordinate descent around the best point
        if calls_made[0] < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.2
            for _ in range(20):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        fx_trial = counted_func(x_trial)
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                if not improved:
                    step *= 0.7
                    if step < 0.01:
                        break
                else:
                    step = max(0.05, step * 0.9)

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = counted_func(best_x)

        return best_x, best_f
