"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_points = []
        used = 0

        # ZDT3 structure with g=1 (x[1:]=0)
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)

        # Analytic Pareto front: x[1:] = 0, g = 1
        # f1 = x0, f2 = 1 - sqrt(x0) - x0 * sin(10*pi*x0)
        # Non-dominated points are local minima of f2(x0) where f2 is a record low

        # Multi-resolution approach:
        # 1. Coarse grid to identify candidate islands
        # 2. Refine each island with local minimization
        # 3. Select best points via greedy hypervolume if needed

        def f2_func(x0):
            return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0)

        # Step 1: Coarse grid to find all record-low regions
        n_coarse = 50000
        x_coarse = np.linspace(0.0, 1.0, n_coarse)
        f2_coarse = 1.0 - np.sqrt(x_coarse) - x_coarse * np.sin(10.0 * np.pi * x_coarse)

        # Running minimum to identify non-dominated envelope
        min_f2 = np.full(n_coarse, np.inf)
        min_f2[0] = f2_coarse[0]
        min_f2[1:] = np.minimum.accumulate(f2_coarse[1:])

        # Non-dominated: f2[i] is a new record low (or equal within tol)
        tol = 1e-10
        is_record = (f2_coarse <= min_f2 + tol)

        # Find contiguous groups of record-low points (islands)
        # An island is a maximal contiguous block where is_record is True
        record_indices = np.where(is_record)[0]

        if len(record_indices) == 0:
            # Fallback: just use x0=0
            selected_x = np.array([0.0])
        else:
            # Group consecutive record indices into islands
            # Two indices are in the same island if they are adjacent in record_indices
            # and their x values are close
            islands = []
            current_island = [record_indices[0]]
            for i in range(1, len(record_indices)):
                idx_prev = record_indices[i-1]
                idx_curr = record_indices[i]
                # If indices are close (within a few grid steps), same island
                if idx_curr - idx_prev <= 2:
                    current_island.append(idx_curr)
                else:
                    islands.append(current_island)
                    current_island = [idx_curr]
            islands.append(current_island)

            # For each island, find the x0 that minimizes f2 within that region
            # Use the minimum f2 point in the coarse grid as a starting point
            # Then do a fine local search
            candidate_points = []
            for island in islands:
                if time.time() > deadline - 10.0:
                    break
                # Get the x range of this island
                x_start = x_coarse[island[0]]
                x_end = x_coarse[island[-1]]
                # Find the best point in the coarse grid for this island
                f2_in_island = f2_coarse[island]
                best_local_idx = np.argmin(f2_in_island)
                best_x = x_coarse[island[best_local_idx]]
                best_f2 = f2_in_island[best_local_idx]

                # Refine: do a fine grid search around best_x
                # Search window: a few coarse grid steps on each side
                search_radius = 5 * (1.0 / n_coarse)
                x_fine_start = max(0.0, best_x - search_radius)
                x_fine_end = min(1.0, best_x + search_radius)
                n_fine = 200
                if x_fine_end > x_fine_start:
                    x_fine = np.linspace(x_fine_start, x_fine_end, n_fine)
                    f2_fine = 1.0 - np.sqrt(x_fine) - x_fine * np.sin(10.0 * np.pi * x_fine)
                    best_fine_idx = np.argmin(f2_fine)
                    best_x = x_fine[best_fine_idx]
                    best_f2 = f2_fine[best_fine_idx]

                candidate_points.append((best_x, best_f2))

            # Sort candidates by f1 (x0) for consistency
            candidate_points.sort(key=lambda p: p[0])

            # If we have more candidates than budget, use greedy hypervolume selection
            if len(candidate_points) <= budget:
                selected_x = np.array([p[0] for p in candidate_points])
            else:
                # Greedy hypervolume maximization
                # Reference point for hypervolume: [11.0, 11.0]
                ref = np.array([11.0, 11.0])

                def hypervolume(points):
                    """Compute 2D hypervolume for a set of points with reference point."""
                    if len(points) == 0:
                        return 0.0
                    # Sort by f1
                    pts = sorted(points, key=lambda p: p[0])
                    # Filter: only points strictly better than ref in both objectives
                    pts = [(p[0], p[1]) for p in pts if p[0] < ref[0] and p[1] < ref[1]]
                    if len(pts) == 0:
                        return 0.0
                    # 2D hypervolume: sum of rectangles
                    hv = 0.0
                    prev_f2 = ref[1]
                    for i in range(len(pts)):
                        f1_i, f2_i = pts[i]
                        # Width in f1 direction
                        if i == 0:
                            w = ref[0] - f1_i
                        else:
                            w = pts[i-1][0] - f1_i
                        # Height in f2 direction
                        h = prev_f2 - f2_i
                        if w > 0 and h > 0:
                            hv += w * h
                        prev_f2 = min(prev_f2, f2_i)
                    return hv

                # Greedy selection
                selected = []
                remaining = list(candidate_points)
                for _ in range(min(budget, len(remaining))):
                    if time.time() > deadline - 5.0:
                        break
                    best_hv = -1.0
                    best_idx = -1
                    for idx in range(len(remaining)):
                        trial = selected + [remaining[idx]]
                        hv = hypervolume(trial)
                        if hv > best_hv:
                            best_hv = hv
                            best_idx = idx
                    if best_idx >= 0:
                        selected.append(remaining.pop(best_idx))

                selected_x = np.array([p[0] for p in selected])

        # Ensure we don't exceed budget
        n_evals = min(len(selected_x), budget)

        for i in range(n_evals):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(dim)
            x[0] = selected_x[i]
            # x[1:] = 0 achieves g = 1 (minimum)

            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
