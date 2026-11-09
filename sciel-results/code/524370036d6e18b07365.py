"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a deterministic coarse-to-fine grid search followed by
    Nelder-Mead local refinement. Designed for Rastrigin's many local minima:
    the grid systematically maps the landscape, then NM refines the best basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 315.0

        best_x = None
        best_fx = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_fx, calls
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse-to-fine grid search (80% of budget)
        grid_budget = int(budget * 0.80)

        # Level 1: Very coarse grid (3 points per dimension = 3^5 = 243 points)
        # Points: -4, 0, 4
        coarse_points = np.array([-4.0, 0.0, 4.0])
        grid1 = np.array(np.meshgrid(*([coarse_points] * dim)), dtype=np.float64)
        grid1 = grid1.reshape(-1, dim)

        for pt in grid1:
            if calls >= grid_budget or time.time() > deadline - 10:
                break
            evaluate(pt)

        # Level 2: Medium grid (5 points per dimension = 5^5 = 3125 points)
        # Points: -4, -2, 0, 2, 4
        medium_points = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        grid2 = np.array(np.meshgrid(*([medium_points] * dim)), dtype=np.float64)
        grid2 = grid2.reshape(-1, dim)

        for pt in grid2:
            if calls >= grid_budget or time.time() > deadline - 10:
                break
            evaluate(pt)

        # Level 3: Fine grid around the best region found so far
        if best_x is not None and calls < grid_budget:
            # Create a fine grid centered on best_x with extent 2.0
            fine_extent = 2.0
            fine_points = np.linspace(-fine_extent, fine_extent, 5) + best_x
            fine_points = np.clip(fine_points, -5, 5)
            grid3 = np.array(np.meshgrid(*([fine_points] * dim)), dtype=np.float64)
            grid3 = grid3.reshape(-1, dim)

            for pt in grid3:
                if calls >= grid_budget or time.time() > deadline - 10:
                    break
                evaluate(pt)

        # Phase 2: Nelder-Mead local polish (remaining 20% of budget)
        if best_x is not None and calls < budget - 5 and time.time() < deadline - 5:
            x_nm = best_x.copy()

            # Build initial simplex
            simplex = [x_nm.copy()]
            fvals = [best_fx]
            for i in range(dim):
                pert = x_nm.copy()
                delta = 0.1 if x_nm[i] < 4.9 else -0.1
                pert[i] += delta
                pert = np.clip(pert, -5, 5)
                simplex.append(pert)
                fvals.append(func(pert))
                calls += 1

            # Nelder-Mead iterations
            max_nm_iters = min(80, budget - calls)
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + 2 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
