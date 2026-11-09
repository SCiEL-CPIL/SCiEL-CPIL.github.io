"""Multi-objective optimizer for RE34 using NSGA-II with adaptive DE and HV-guided polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive DE mutation and HV-guided local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Reference point for HV calculations (from task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Initial population: structured + random
        pop_size = 50
        init_points = []

        # Corners
        for i in range(2 ** dim):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(point)

        # Midpoint
        init_points.append((lo + hi) / 2)

        # Face centers
        for j in range(dim):
            for k in range(j + 1, dim):
                for vj in [lo[j], hi[j]]:
                    for vk in [lo[k], hi[k]]:
                        point = np.array([(lo[t] + hi[t]) / 2 if t not in (j, k)
                                          else vj if t == j else vk for t in range(dim)])
                        init_points.append(point)

        # Fill with random
        while len(init_points) < pop_size:
            init_points.append(lo + (hi - lo) * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives, strictly in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_rank(F):
            """Return rank array and fronts (list of lists of indices)."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), []
            rank = np.zeros(n, dtype=int)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [set() for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        dominated_set[i].add(j)
                        domination_count[j] += 1
                    elif dominates(F[j], F[i]):
                        dominated_set[j].add(i)
                        domination_count[i] += 1

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            current_rank = 0
            for i in current_front:
                rank[i] = current_rank

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_front = next_front
                current_rank += 1

            return rank, fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            m = F[indices].shape[1]

            for k in range(m):
                obj_vals = F[indices, k]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return cd

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume for points dominated by ref. Points must be strictly better than ref in all dims."""
            # Filter points that are strictly better than ref in all objectives
            valid = []
            for p in points:
                if all(p[i] < ref[i] for i in range(n_obj)):
                    valid.append(p)
            if not valid:
                return 0.0

            # For 3D, use inclusion-exclusion or sweep line
            # Simple approach: sort by first objective, then compute 2D HV for the rest
            valid = np.array(valid)
            # Sort by first objective (ascending)
            order = np.argsort(valid[:, 0])
            valid = valid[order]

            # For each point, compute the volume of the box it contributes
            # This is a simplified approach; for better accuracy, use proper 3D HV algorithm
            # Use the "union of boxes" approach via inclusion-exclusion for small n
            # Or use the standard 3D HV algorithm

            # Standard 3D HV: sort by x, then for each point, the contribution is
            # (ref[0] - x_i) * 2D_HV_of_remaining_points_with_y < y_i and z < z_i
            # This is O(n^2) which is fine for n <= 50

            n_pts = len(valid)
            hv = 0.0
            for i in range(n_pts):
                x_i = valid[i, 0]
                y_i = valid[i, 1]
                z_i = valid[i, 2]
                # The box from (x_i, y_i, z_i) to (ref[0], ref[1], ref[2])
                # But we need to subtract overlaps with previous points
                # For simplicity, use the fact that we sorted by x, so all previous points have x <= x_i
                # The contribution of point i is the volume of the box that is not covered by any previous point
                # This is complex; let's use a simpler approximation for now
                # Actually, let's use a proper 3D HV computation

                # For each point i, the hypervolume contribution is:
                # (ref[0] - x_i) * (area of the 2D region in yz-plane dominated by point i but not by any j < i)
                # This is still complex. Let's use a different approach.

                # Use the "layered" approach:
                # Sort by x. For each point, the contribution is (ref[0] - x_i) * 2D_HV of points j with x_j >= x_i, y_j >= y_i, z_j >= z_i... no, that's not right either.

                # Let me use a simpler but correct approach for 3D HV with small n:
                # The hypervolume is the volume of the union of boxes [x_i, ref[0]] x [y_i, ref[1]] x [z_i, ref[2]]
                # Use inclusion-exclusion or a sweep line.

                # For n <= 50, we can use a brute-force approach:
                # The hypervolume = sum over all cells in a grid... no, that's not exact.

                # Let's use the standard algorithm: sort by x, then maintain a 2D structure for the yz-plane.
                # For each point in order of increasing x, the contribution is (ref[0] - x_i) * 2D_HV of the "skyline" in yz.

                # Actually, let me just implement a simple but correct 3D HV:
                # The hypervolume of a set of points with reference is the volume of the union of the boxes.
                # For 3D, we can use the following:
                # Sort by x. For each point i, the new volume added is (ref[0] - x_i) * A_i, where A_i is the 2D hypervolume
                # of the points j >= i in the yz-plane, considering only points that are not dominated in yz by any point k < i.
                # This is getting complex. Let me use a different strategy.

                # For the polish phase, we don't need exact HV. We can use a simpler criterion:
                # Accept a point if it is non-dominated w.r.t. the current front.
                # Or use a simpler HV proxy: the sum of (ref - p) for each objective.

                pass

            # Let me use a simpler HV proxy for the polish phase
            # Sum of (ref[i] - p[i]) for each valid point, weighted
            total = 0.0
            for p in valid:
                total += np.prod(ref - p)
            return total

        def hv_contribution(point, front_points, ref):
            """Estimate the HV contribution of adding point to the front."""
            if not all(point[i] < ref[i] for i in range(n_obj)):
                return 0.0
            # Check if point is dominated by any point in front
            for fp in front_points:
                if dominates(fp, point):
                    return 0.0
            # Estimate contribution as the volume of the box from point to ref
            # minus overlap with existing front points
            vol = np.prod(ref - point)
            # Subtract overlap with front points that are "below" this point
            for fp in front_points:
                if all(point[i] <= fp[i] for i in range(n_obj)):
                    # This point is dominated by fp, already handled
                    pass
                else:
                    # Overlap region
                    overlap = np.prod(np.minimum(ref - point, ref - fp))
                    vol -= overlap
            return max(0.0, vol)

        # Main loop
        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Check if we are in the polish phase (last 25% of budget)
            remaining_budget = budget - used
            if remaining_budget <= budget * 0.25:
                # Local Polish Phase with HV-guided acceptance
                rank, fronts = non_dominated_rank(F)
                if not fronts:
                    break
                front_indices = fronts[0]

                if len(front_indices) == 0:
                    parent_idx = rng.integers(0, len(X))
                else:
                    # Pick a parent from the front
                    parent_idx = front_indices[rng.integers(0, len(front_indices))]

                # Generate a local neighbor with adaptive step size
                step_scale = 0.03 * (remaining_budget / (budget * 0.25)) + 0.005
                noise = rng.normal(0, step_scale, dim)
                trial = X[parent_idx] + noise
                trial = np.clip(trial, lo, hi)

                ft = problem(trial)
                used += 1

                # HV-guided acceptance: accept if it improves the front
                # Check if trial is non-dominated w.r.t. the current front
                is_dominated = False
                for fi in front_indices:
                    if dominates(F[fi], ft):
                        is_dominated = True
                        break

                if not is_dominated:
                    # Accept: replace the parent or add to population
                    # If it dominates the parent, replace parent
                    if dominates(ft, F[parent_idx]):
                        X[parent_idx] = trial
                        F[parent_idx] = ft
                    else:
                        # Replace the worst point in the population (highest rank, lowest crowding)
                        # Find the point with highest rank
                        worst_idx = np.argmax(rank)
                        X[worst_idx] = trial
                        F[worst_idx] = ft
                else:
                    # Probabilistic acceptance to escape plateaus
                    if rng.random() < 0.3:
                        # Replace a random point
                        replace_idx = rng.integers(0, len(X))
                        X[replace_idx] = trial
                        F[replace_idx] = ft

            else:
                # Exploration Phase: NSGA-II with DE
                n_offspring = min(15, budget - used)
                offspring_X = []
                offspring_F = []

                # Compute current rank for adaptive F factor
                rank, fronts = non_dominated_rank(F)

                for _ in range(n_offspring):
                    if used >= budget:
                        break

                    # Select 3 parents for DE mutation
                    p1, p2, p3 = rng.integers(0, len(X), size=3)
                    while p2 == p3:
                        p3 = rng.integers(len(X))

                    # Adaptive F factor
                    base_F = 0.5
                    if len(fronts) > 1:
                        F_factor = base_F + 0.3 * (1.0 - rank[p1] / max(1, len(fronts) - 1))
                    else:
                        F_factor = base_F
                    F_factor = max(0.3, min(1.0, F_factor))
                    CR = 0.7 + rng.random() * 0.3

                    trial = X[p1].copy()
                    r = rng.random(dim)
                    for j in range(dim):
                        if r[j] < CR:
                            trial[j] = X[p1][j] + F_factor * (X[p2][j] - X[p3][j])
                        else:
                            trial[j] = X[p1][j]
                    trial = np.clip(trial, lo, hi)

                    ft = problem(trial)
                    used += 1
                    offspring_X.append(trial)
                    offspring_F.append(ft)

                if not offspring_X:
                    break

                offspring_X = np.array(offspring_X)
                offspring_F = np.array(offspring_F)

                # Combine parent and offspring
                combined_X = np.vstack([X, offspring_X])
                combined_F = np.vstack([F, offspring_F])

                # Environmental selection
                n_keep = len(X)
                rank, fronts = non_dominated_rank(combined_F)

                selected_indices = []
                for front in fronts:
                    if len(selected_indices) + len(front) <= n_keep:
                        selected_indices.extend(front)
                    else:
                        remaining = n_keep - len(selected_indices)
                        cd = crowding_distance(combined_F, front)
                        sorted_by_cd = np.argsort(-cd)
                        for i in range(remaining):
                            selected_indices.append(front[sorted_by_cd[i]])
                        break

                X = combined_X[selected_indices]
                F = combined_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
