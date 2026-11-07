"""Multi-start Nelder-Mead optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time and budget guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead implementation
        def nelder_mead(start, max_iter):
            nonlocal calls_made, best_x, best_f, budget
            dim = self.dim
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = start
            for i in range(1, dim + 1):
                simplex[i] = start.copy()
                simplex[i, i - 1] += 0.1

            fvals = np.array([eval_func(simplex[i]) for i in range(dim + 1)])
            calls_made += 0  # already counted in eval_func

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_func(x_r)
                if calls_made >= budget:
                    break

                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_func(x_e)
                    if calls_made >= budget:
                        break
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
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = eval_func(x_c)
                    if calls_made >= budget:
                        break
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = eval_func(simplex[i])
                            if calls_made >= budget:
                                return simplex[0], fvals[0]

            # Return best point from simplex
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Multi-start strategy: use remaining budget for multiple restarts
        # Allocate budget: initial exploration + local refinement
        # Strategy: divide budget into chunks, each gets a random start + Nelder-Mead

        # Use a mix of random starts and best-found refinement
        # First, do a quick random search to find good basins
        n_random = min(20, budget // 10)
        for i in range(n_random):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)

        # Now do multiple Nelder-Mead runs from best points found
        # We'll do restarts with shrinking step size
        remaining_budget = budget - calls_made
        if remaining_budget > 10:
            # Number of restarts based on remaining budget
            n_restarts = min(10, max(1, remaining_budget // 50))
            budget_per_restart = remaining_budget // n_restarts

            for r in range(n_restarts):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Start from best found or random
                if best_x is not None and r < n_restarts // 2:
                    # Perturb best found
                    x0 = best_x + rng.standard_normal(dim) * 0.5
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)

                # Run Nelder-Mead with limited iterations
                nm_iter = min(budget_per_restart - (dim + 1), 100)
                if nm_iter < 5:
                    break
                nelder_mead(x0, nm_iter)

        # Final refinement if budget remains
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 10:
                nm_iter = min(remaining - (dim + 1), 200)
                if nm_iter >= 5:
                    nelder_mead(best_x, nm_iter)

        return best_x, best_f
# EVOLVE-BLOCK-END
