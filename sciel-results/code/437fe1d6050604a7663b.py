"""Grid-based decomposition with local simplex polish for RE21 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Grid-decomposition multi-objective optimizer with local simplex refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def clip(x):
            return np.clip(x, lo, hi)

        def is_deadline():
            return (time.time() - t0) > deadline

        # Phase 1: Coarse grid sampling for uniform coverage
        # 4D space: use 3 points per dimension = 81 grid points
        grid_pts_per_dim = 3
        n_grid = grid_pts_per_dim ** dim  # 81 for 4D

        # Generate grid points (deterministic, no randomness needed for grid)
        grid_axes = [np.linspace(lo[i], hi[i], grid_pts_per_dim) for i in range(dim)]
        grid_points = []
        for a0 in grid_axes[0]:
            for a1 in grid_axes[1]:
                for a2 in grid_axes[2]:
                    for a3 in grid_axes[3]:
                        grid_points.append(np.array([a0, a1, a2, a3]))
        grid_points = np.array(grid_points)

        # Evaluate grid points
        all_F = []
        all_X = []
        for i in range(n_grid):
            if is_deadline():
                break
            x = grid_points[i]
            f = np.array(problem(x), dtype=np.float64)
            all_F.append(f)
            all_X.append(x)

        # Phase 2: Local simplex refinement on non-dominated grid points
        # Find non-dominated points from grid
        F_arr = np.array(all_F)
        X_arr = np.array(all_X)

        def dominated_mask(F):
            """Return boolean mask: True if point i is dominated by any other point."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return dominated

        non_dom_mask = ~dominated_mask(F_arr)
        non_dom_X = X_arr[non_dom_mask]
        non_dom_F = F_arr[non_dom_mask]

        # Use remaining budget for local optimization
        used = len(all_F)
        remaining = budget - used

        # Select top candidates for local refinement
        # Prioritize points that are well-spread on the Pareto front
        if len(non_dom_X) > 0:
            # Rank by hypervolume contribution (approximate by sum of objectives)
            # Pick the most diverse set
            n_refine = min(len(non_dom_X), max(5, remaining // 15))

            # Simple diversity selection: spread across objective space
            if len(non_dom_F) <= n_refine:
                refine_idx = np.arange(len(non_dom_X))
            else:
                # Select points that are far apart in objective space
                refine_idx = []
                # Start with point with best sum of objectives
                sums = np.sum(non_dom_F, axis=1)
                best_idx = np.argmin(sums)
                refine_idx.append(best_idx)
                while len(refine_idx) < n_refine:
                    # Find point farthest from already selected
                    best_dist = -1
                    best_i = -1
                    for i in range(len(non_dom_X)):
                        if i in refine_idx:
                            continue
                        min_dist = np.min(
                            [np.linalg.norm(non_dom_F[i] - non_dom_F[j]) for j in refine_idx]
                        )
                        if min_dist > best_dist:
                            best_dist = min_dist
                            best_i = i
                    if best_i >= 0:
                        refine_idx.append(best_i)
                    else:
                        break

            # Local Nelder-Mead simplex refinement on each selected point
            # Each refinement uses a few evaluations
            evals_per_refine = 12
            n_refine = min(n_refine, remaining // max(1, evals_per_refine))

            for idx in refine_idx:
                if is_deadline():
                    break
                if used >= budget:
                    break

                x0 = non_dom_X[idx].copy()

                # Define scalarized objective (weighted sum) for local optimization
                # Try a few weightings to explore different parts of the front
                weights = [
                    np.array([1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.5, 0.5]),
                    np.array([0.8, 0.2]),
                    np.array([0.2, 0.8]),
                ]

                # Pick weight based on the objective values at x0
                f0 = non_dom_F[idx]
                # Use the point's own relative position as weight hint
                if len(non_dom_F) > 1:
                    f_min = np.min(non_dom_F, axis=0)
                    f_max = np.max(non_dom_F, axis=0)
                    f_range = f_max - f_min + 1e-12
                    rel_pos = (f0 - f_min) / f_range
                    w = 1.0 - rel_pos  # Points with low f1 get high weight on f1
                    w = np.clip(w, 0.1, 0.9)
                    weights = [w]
                else:
                    weights = [np.array([0.5, 0.5])]

                for w in weights:
                    if is_deadline() or used >= budget:
                        break

                    def scalar_obj(x, w=w):
                        fx = problem(x)
                        return float(np.dot(w, fx))

                    # Nelder-Mead with limited evaluations
                    try:
                        result = minimize(
                            scalar_obj,
                            x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 30,
                                'xatol': 1e-4,
                                'fatol': 1e-6,
                                'adaptive': True,
                            }
                        )
                        # Count evaluations used (approximate)
                        n_evals = result.nfev if hasattr(result, 'nfev') else evals_per_refine
                        n_evals = min(n_evals, budget - used)
                        used += n_evals

                        x_best = clip(result.x)
                        f_best = np.array(problem(x_best), dtype=np.float64)
                        all_X.append(x_best)
                        all_F.append(f_best)
                    except Exception:
                        pass

        # Phase 3: Use any remaining budget for additional random sampling
        # focused near the best region found
        if used < budget and not is_deadline():
            # Find the best region
            if len(all_F) > 0:
                F_final = np.array(all_F)
                # Find non-dominated again
                nd_mask = ~dominated_mask(F_final)
                nd_F = F_final[nd_mask]
                if len(nd_F) > 0:
                    # Sample around non-dominated points
                    rng = np.random.default_rng(42)
                    while used < budget and not is_deadline():
                        # Pick a random non-dominated point and perturb
                        idx = rng.integers(len(nd_F))
                        base_F = nd_F[idx]
                        # Find corresponding X
                        # (approximate: just sample near the center of bounds)
                        x = lo + (hi - lo) * rng.random(dim)
                        # Bias toward the center of the current non-dominated region
                        center = np.mean(nd_F, axis=0)
                        # Map objective center back to a rough x (heuristic)
                        # Just use uniform sampling in the box
                        f = np.array(problem(x), dtype=np.float64)
                        all_X.append(x)
                        all_F.append(f)
                        used += 1

        # Return the archive (all evaluated points)
        return np.array(all_X)


# EVOLVE-BLOCK-END
