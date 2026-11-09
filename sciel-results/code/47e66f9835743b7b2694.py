"""Rastrigin f3 optimizer exploiting separable periodic structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits the separable periodic structure of Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0

        rng = np.random.default_rng(42)

        # Phase 1: Coordinate-wise 1D grid search to find the global basin
        # For each dimension, do a coarse 1D scan to find the best coordinate
        # given the other coordinates are at their current best values.

        # Start with a reasonable initial guess
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Coarse grid: for each dimension, try values from -5 to 5 in steps of ~1.0
        # Then refine around the best found
        grid_coarse = np.linspace(-5.0, 5.0, 21)  # 21 points per dim
        grid_fine = np.linspace(-5.0, 5.0, 51)    # 51 points per dim

        # First pass: coarse scan
        for _pass in range(3):  # 3 passes of coordinate descent
            if calls >= budget - 100 or time.time() > deadline - 10:
                break
            for d in range(dim):
                if calls >= budget - 100 or time.time() > deadline - 10:
                    break
                # Determine grid for this dimension
                if _pass == 0:
                    grid = grid_coarse
                elif _pass == 1:
                    # Fine grid around current best
                    center = x[d]
                    span = 2.0 if _pass == 1 else 0.5
                    grid = np.linspace(max(-5, center - span), min(5, center + span), 21)
                else:
                    center = x[d]
                    span = 0.3
                    grid = np.linspace(max(-5, center - span), min(5, center + span), 15)

                best_val = fx
                best_coord = x[d]
                for g in grid:
                    if calls >= budget - 100 or time.time() > deadline - 10:
                        break
                    trial = x.copy()
                    trial[d] = g
                    trial = np.clip(trial, -5, 5)
                    ft = func(trial)
                    calls += 1
                    if ft < best_val:
                        best_val = ft
                        best_coord = g

                x[d] = best_coord
                fx = best_val
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 2: Fine coordinate-wise refinement
        for _pass in range(5):
            if calls >= budget - 50 or time.time() > deadline - 10:
                break
            for d in range(dim):
                if calls >= budget - 50 or time.time() > deadline - 10:
                    break
                center = x[d]
                # Very fine local grid
                span = 0.1 * (0.7 ** _pass) + 0.005
                grid = np.linspace(max(-5, center - span), min(5, center + span), 11)
                best_val = fx
                best_coord = x[d]
                for g in grid:
                    if calls >= budget - 50 or time.time() > deadline - 10:
                        break
                    trial = x.copy()
                    trial[d] = g
                    trial = np.clip(trial, -5, 5)
                    ft = func(trial)
                    calls += 1
                    if ft < best_val:
                        best_val = ft
                        best_coord = g

                x[d] = best_coord
                fx = best_val
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 3: Nelder-Mead polish on the full vector
        if calls < budget - 10 and time.time() < deadline - 5:
            x_nm = best_x.copy()
            simplex = [x_nm.copy()]
            for i in range(dim):
                pert = x_nm.copy()
                pert[i] += 0.01
                simplex.append(np.clip(pert, -5, 5))

            fvals = []
            for s in simplex:
                if calls >= budget:
                    break
                fvals.append(func(s))
                calls += 1

            if len(fvals) == dim + 1:
                max_nm_iter = min(200, budget - calls - 5)
                for _ in range(max_nm_iter):
                    if time.time() > deadline - 5 or calls >= budget:
                        break

                    idx = np.argsort(fvals)
                    simplex = [simplex[i] for i in idx]
                    fvals = [fvals[i] for i in idx]

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                    fr = func(xr)
                    calls += 1
                    if calls >= budget:
                        break

                    if fr < fvals[0]:
                        if calls < budget:
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
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        if calls < budget:
                            xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                            fc = func(xc)
                            calls += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, dim + 1):
                                if calls >= budget:
                                    break
                                simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                                fvals[i] = func(simplex[i])
                                calls += 1

                    if fvals[0] < best_fx:
                        best_fx = fvals[0]
                        best_x = simplex[0].copy()

        # Phase 4: Final fine-tune with very small steps around best
        if calls < budget - 5 and time.time() < deadline - 5:
            step = 0.001
            for _ in range(10):
                if calls >= budget - 5 or time.time() > deadline - 5:
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        if calls >= budget - 5 or time.time() > deadline - 5:
                            break
                        trial = best_x.copy()
                        trial[d] += sign * step
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        calls += 1
                        if ft < best_fx:
                            best_fx = ft
                            best_x = trial.copy()
                            improved = True
                if not improved:
                    break
                step *= 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
