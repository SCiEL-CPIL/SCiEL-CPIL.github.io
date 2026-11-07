"""Rewritten optimizer: grid-based multi-start with Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based multi-start with Nelder-Mead local optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Count function calls
        call_count = [0]

        def eval_func(x):
            call_count[0] += 1
            return float(func(x))

        # Phase 1: Evaluate a sparse 3^5 grid to find promising basins
        # Use a 3-point grid in each dimension: -4, 0, 4 (within [-5,5])
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = np.array(np.meshgrid(*([grid_vals] * dim))).reshape(dim, -1).T

        # If budget allows, evaluate all grid points
        n_grid = len(grid_points)
        grid_budget = min(n_grid, budget // 2)  # Use at most half budget for grid

        # If we can't evaluate all grid points, sample a subset
        if n_grid <= grid_budget:
            points_to_eval = grid_points
        else:
            # Stratified sampling of grid points
            indices = np.linspace(0, n_grid - 1, grid_budget, dtype=int)
            points_to_eval = grid_points[indices]

        # Evaluate grid points
        grid_results = []
        for pt in points_to_eval:
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            fx = eval_func(pt)
            grid_results.append((fx, pt.copy()))

        # Sort by fitness and pick top 5 (or fewer if we ran out of budget)
        grid_results.sort(key=lambda x: x[0])
        top_k = min(5, len(grid_results))
        if top_k == 0:
            # Fallback: just return the best we have
            best_x = np.zeros(dim)
            best_f = float('inf')
            return best_x, best_f

        # Phase 2: Nelder-Mead local optimization from each top point
        best_x = grid_results[0][1].copy()
        best_f = grid_results[0][0]

        remaining_budget = budget - call_count[0]
        per_local = max(1, remaining_budget // top_k)

        for i in range(top_k):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

            x0 = grid_results[i][1]
            f0 = grid_results[i][0]

            # Nelder-Mead implementation
            x = x0.copy()
            fx = f0

            # Initialize simplex
            simplex = [x.copy()]
            simplex_f = [fx]

            # Create initial simplex with small perturbations
            for d in range(dim):
                xp = x.copy()
                xp[d] += 0.1
                xp = np.clip(xp, -5, 5)
                fp = eval_func(xp)
                simplex.append(xp)
                simplex_f.append(fp)

            # Track calls for this local optimization
            local_calls = 0
            max_local_calls = per_local - (dim + 1)  # account for simplex initialization

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            while local_calls < max_local_calls and call_count[0] < budget:
                if time.time() > deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[j] for j in order]
                simplex_f = [simplex_f[j] for j in order]

                best_f_simplex = simplex_f[0]
                worst_idx = dim  # last element
                worst_f = simplex_f[worst_idx]
                second_worst_f = simplex_f[dim - 1]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:dim], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[worst_idx])
                xr = np.clip(xr, -5, 5)
                fr = eval_func(xr)
                local_calls += 1

                if fr < simplex_f[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[worst_idx])
                    xe = np.clip(xe, -5, 5)
                    fe = eval_func(xe)
                    local_calls += 1

                    if fe < fr:
                        simplex[worst_idx] = xe
                        simplex_f[worst_idx] = fe
                    else:
                        simplex[worst_idx] = xr
                        simplex_f[worst_idx] = fr
                elif fr < second_worst_f:
                    simplex[worst_idx] = xr
                    simplex_f[worst_idx] = fr
                else:
                    # Contraction
                    if fr < worst_f:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = eval_func(xc)
                        local_calls += 1

                        if fc <= fr:
                            simplex[worst_idx] = xc
                            simplex_f[worst_idx] = fc
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                simplex_f[j] = eval_func(simplex[j])
                                local_calls += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[worst_idx])
                        xc = np.clip(xc, -5, 5)
                        fc = eval_func(xc)
                        local_calls += 1

                        if fc < worst_f:
                            simplex[worst_idx] = xc
                            simplex_f[worst_idx] = fc
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                simplex_f[j] = eval_func(simplex[j])
                                local_calls += 1

                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
