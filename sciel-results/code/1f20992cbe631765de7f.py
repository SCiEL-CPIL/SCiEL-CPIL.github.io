"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware multi-objective search for WFG9.

    Exploits WFG9's known separable structure: the first three variables
    control the position along the Pareto front, while variables 4-12
    control a 'distance' term that should be driven to zero. We use a
    hybrid approach: grid/sample the first 3 variables densely and optimize
    the remaining 9 variables to minimize the distance term via local search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # WFG9 structure: first 3 vars are critical for Pareto front shape.
        # Vars 4-12 contribute to a distance-to-front term.
        # Strategy: sample first 3 vars on a grid, then locally optimize vars 4-12.

        # Phase 1: Coarse grid over first 3 variables
        # Use a moderate grid to stay within budget
        n_grid = 8  # 8^3 = 512 points, but we'll subsample
        budget_used = 0
        max_budget = self.budget

        results = []  # list of (x, f)

        def evaluate(x):
            nonlocal budget_used
            if budget_used >= max_budget:
                return None
            f = problem(x)
            budget_used += 1
            return f

        # Phase 1: Sample first 3 variables
        # For WFG9, the Pareto front is in the first 3 dimensions essentially.
        # Let's create a grid over x[0], x[1], x[2]
        grid_pts_3d = []
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    if budget_used >= max_budget:
                        break
                    x3 = np.array([
                        lo[0] + (hi[0] - lo[0]) * i / (n_grid - 1),
                        lo[1] + (hi[1] - lo[1]) * j / (n_grid - 1),
                        lo[2] + (hi[2] - lo[2]) * k / (n_grid - 1),
                    ])
                    grid_pts_3d.append(x3)
                    if budget_used >= max_budget or time.time() > deadline:
                        break
                if budget_used >= max_budget or time.time() > deadline:
                    break
            if budget_used >= max_budget or time.time() > deadline:
                break

        # For each 3D point, try to optimize the remaining 9 variables
        # using a few random restarts + local refinement
        n_local = 5  # number of local optimization attempts per 3D point

        for x3 in grid_pts_3d:
            if budget_used >= max_budget or time.time() > deadline:
                break

            # Initialize remaining vars randomly
            best_f = None
            best_x_rem = None

            for attempt in range(n_local):
                if budget_used >= max_budget or time.time() > deadline:
                    break
                # Random initialization for vars 4-12
                x_rem = lo[3:] + (hi[3:] - lo[3:]) * np.random.random(n - 3)
                x = np.concatenate([x3, x_rem])
                f = evaluate(x)
                if f is None:
                    break
                if best_f is None or np.sum(f) < np.sum(best_f):
                    best_f = f
                    best_x_rem = x_rem.copy()

                # Local search: perturb remaining variables
                for step in range(10):
                    if budget_used >= max_budget or time.time() > deadline:
                        break
                    # Gaussian perturbation, shrinking step size
                    scale = 0.1 * (hi[3:] - lo[3:]) * (0.5 ** step)
                    x_rem_new = best_x_rem + scale * np.random.normal(size=n - 3)
                    x_rem_new = np.clip(x_rem_new, lo[3:], hi[3:])
                    x_new = np.concatenate([x3, x_rem_new])
                    f_new = evaluate(x_new)
                    if f_new is None:
                        break
                    if np.sum(f_new) < np.sum(best_f):
                        best_f = f_new
                        best_x_rem = x_rem_new.copy()

            if best_f is not None:
                x_full = np.concatenate([x3, best_x_rem])
                results.append((x_full, best_f))

        # Phase 2: If we have budget left, do a global random search + local refinement
        if budget_used < max_budget and time.time() < deadline:
            n_random = min(50, max_budget - budget_used)
            for _ in range(n_random):
                if budget_used >= max_budget or time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(n)
                f = evaluate(x)
                if f is None:
                    break
                results.append((x, f))

                # Quick local search
                for step in range(5):
                    if budget_used >= max_budget or time.time() > deadline:
                        break
                    scale = 0.05 * (hi - lo) * (0.5 ** step)
                    x_new = x + scale * np.random.normal(size=n)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = evaluate(x_new)
                    if f_new is None:
                        break
                    if np.sum(f_new) < np.sum(f):
                        x = x_new
                        f = f_new
                        results.append((x, f))

        # Phase 3: Final polish on best points
        if budget_used < max_budget and time.time() < deadline:
            # Find the best few points by sum of objectives
            if results:
                sorted_results = sorted(results, key=lambda t: np.sum(t[1]))
                for x_best, f_best in sorted_results[:10]:
                    if budget_used >= max_budget or time.time() > deadline:
                        break
                    # High-precision local search
                    for step in range(20):
                        if budget_used >= max_budget or time.time() > deadline:
                            break
                        scale = 0.01 * (hi - lo) * (0.8 ** step)
                        x_new = x_best + scale * np.random.normal(size=n)
                        x_new = np.clip(x_new, lo, hi)
                        f_new = evaluate(x_new)
                        if f_new is None:
                            break
                        if np.sum(f_new) < np.sum(f_best):
                            x_best = x_new
                            f_best = f_new
                            results.append((x_best, f_best))

        # Return the archive (the return value is ignored, but we return something valid)
        if results:
            xs = np.array([r[0] for r in results])
            return xs
        return np.zeros((1, n))


# EVOLVE-BLOCK-END
