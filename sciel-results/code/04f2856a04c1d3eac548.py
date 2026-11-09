"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using grid-refinement + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1.1, 1.3, 1.1])

        all_x = []
        all_f = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            all_x.append(x.copy())
            all_f.append(f.copy())
            return f

        def get_front():
            if len(all_f) == 0:
                return np.empty((0, n_obj)), np.array([], dtype=int)
            F_arr = np.array(all_f)
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            front_idx = np.where(mask)[0]
            return F_arr[front_idx], front_idx

        # Phase 1: Coarse grid (3^4 = 81 points)
        grid_levels = 3
        grid_points = []
        for i0 in range(grid_levels):
            for i1 in range(grid_levels):
                for i2 in range(grid_levels):
                    for i3 in range(grid_levels):
                        x = np.array([
                            lo[d] + (hi[d] - lo[d]) * (i + 0.5) / grid_levels
                            for d, i in enumerate([i0, i1, i2, i3])
                        ])
                        grid_points.append(x)

        for x in grid_points:
            if used >= budget or time.time() > deadline:
                break
            eval_point(x)

        front_F, front_idx = get_front()

        # Phase 2: Adaptive grid refinement around front points
        # Refine the 4D hypercube by subdividing cells near front points
        # Use a priority queue based on estimated HV gap

        # Build initial cells from the grid
        # Each cell is defined by [lo_cell, hi_cell] in 4D
        cells = []
        step = (hi - lo) / grid_levels
        for i0 in range(grid_levels):
            for i1 in range(grid_levels):
                for i2 in range(grid_levels):
                    for i3 in range(grid_levels):
                        cell_lo = np.array([lo[d] + i * step[d] for d, i in enumerate([i0, i1, i2, i3])])
                        cell_hi = np.array([lo[d] + (i + 1) * step[d] for d, i in enumerate([i0, i1, i2, i3])])
                        # Check if any front point is in this cell
                        has_front = False
                        for fi in front_idx:
                            xf = all_x[fi]
                            if np.all(xf >= cell_lo) and np.all(xf < cell_hi):
                                has_front = True
                                break
                        cells.append((cell_lo, cell_hi, has_front))

        # Phase 3: Iterative refinement - subdivide cells with front points
        # and evaluate new grid points in promising regions
        max_refine_rounds = 5
        for round_idx in range(max_refine_rounds):
            if used >= budget or time.time() > deadline:
                break

            # Find cells that contain front points and are worth refining
            # Priority: cells with front points, larger cells first
            candidate_cells = [(c[0], c[1]) for c in cells if c[2]]
            if not candidate_cells:
                break

            # Sort by cell volume (larger first) to refine coarse areas
            def cell_vol(cell):
                return np.prod(cell[1] - cell[0])

            candidate_cells.sort(key=cell_vol, reverse=True)

            # Subdivide top cells and evaluate midpoints
            n_subdivide = min(4, len(candidate_cells), (budget - used) // 16)
            new_cells = []
            for cell_lo, cell_hi in candidate_cells[:n_subdivide]:
                if used >= budget or time.time() > deadline:
                    break
                # Subdivide into 2^4 = 16 subcells
                mid = (cell_lo + cell_hi) / 2.0
                subcells = []
                for b0 in range(2):
                    for b1 in range(2):
                        for b2 in range(2):
                            for b3 in range(2):
                                s_lo = np.array([
                                    cell_lo[d] if b == 0 else mid[d]
                                    for d, b in enumerate([b0, b1, b2, b3])
                                ])
                                s_hi = np.array([
                                    mid[d] if b == 0 else cell_hi[d]
                                    for d, b in enumerate([b0, b1, b2, b3])
                                ])
                                subcells.append((s_lo, s_hi))
                new_cells.extend(subcells)

                # Evaluate the center of each subcell
                for s_lo, s_hi in subcells:
                    if used >= budget or time.time() > deadline:
                        break
                    center = (s_lo + s_hi) / 2.0
                    eval_point(center)

            # Update cells: replace subdivided cells with their subcells
            subdivided_set = set()
            for cell_lo, cell_hi in candidate_cells[:n_subdivide]:
                subdivided_set.add((tuple(cell_lo), tuple(cell_hi)))

            updated_cells = []
            for c in cells:
                if (tuple(c[0]), tuple(c[1])) in subdivided_set:
                    continue
                updated_cells.append(c)
            updated_cells.extend([(c[0], c[1], False) for c in new_cells])
            cells = updated_cells

            # Update front
            front_F, front_idx = get_front()

            # Mark cells that now contain front points
            for i, (c_lo, c_hi, _) in enumerate(cells):
                has_front = False
                for fi in front_idx:
                    xf = all_x[fi]
                    if np.all(xf >= c_lo) and np.all(xf < c_hi):
                        has_front = True
                        break
                cells[i] = (c_lo, c_hi, has_front)

        # Phase 4: Local polish - use Nelder-Mead from best front points
        front_F, front_idx = get_front()
        if len(front_F) > 0 and used < budget:
            # Select top front points for polishing
            n_polish = min(12, len(front_F), (budget - used) // 10)
            # Sort by sum of objectives (lower is better)
            sums = np.sum(front_F, axis=1)
            sorted_idx = np.argsort(sums)
            polish_indices = sorted_idx[:n_polish]

            for pi in polish_indices:
                if used >= budget or time.time() > deadline:
                    break
                base_x = all_x[front_idx[pi]].copy()
                base_f = all_f[front_idx[pi]].copy()

                # Try to improve each objective direction
                for obj_idx in range(n_obj):
                    if used >= budget or time.time() > deadline:
                        break

                    # Simple line search in the direction that reduces obj_idx
                    # while trying not to worsen other objectives too much
                    best_x = base_x.copy()
                    best_f = base_f.copy()
                    best_sum = np.sum(base_f)

                    for step_size in [0.1, 0.05, 0.02, 0.01]:
                        if used >= budget or time.time() > deadline:
                            break
                        # Try perturbations in each dimension
                        for d in range(dim):
                            for sign in [-1, 1]:
                                trial_x = best_x.copy()
                                trial_x[d] += sign * step_size * (hi[d] - lo[d])
                                trial_x = np.clip(trial_x, lo, hi)
                                trial_f = eval_point(trial_x)
                                if trial_f is None:
                                    break
                                if np.all(trial_f < ref):
                                    # Check if it's non-dominated vs current front
                                    is_nd = True
                                    for fp in front_F:
                                        if np.all(fp <= trial_f) and np.any(fp < trial_f):
                                            is_nd = False
                                            break
                                    if is_nd:
                                        new_sum = np.sum(trial_f)
                                        if new_sum < best_sum:
                                            best_x = trial_x.copy()
                                            best_f = trial_f.copy()
                                            best_sum = new_sum
                                        front_F, front_idx = get_front()
                        if used >= budget or time.time() > deadline:
                            break

        return np.array(all_x) if all_x else np.empty((0, dim))
# EVOLVE-BLOCK-END
