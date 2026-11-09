"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using differential evolution with
    non-dominated sorting and hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size - smaller for more generations within budget
        pop_size = 24

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (given in task)
        ref = np.array([1.1, 1.3, 1.1])

        def non_dominated_sort(F_arr):
            """Returns list of fronts, each a list of indices."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        def crowding_distance(F_arr, indices):
            """Assign crowding distance to given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)

            return dist

        def dominates(a, b):
            """Check if vector a dominates vector b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def compute_hypervolume(points, ref_point):
            """Compute 3D hypervolume of non-dominated points using inclusion-exclusion."""
            # Filter points strictly better than reference
            valid = [p for p in points if np.all(p < ref_point)]
            if not valid:
                return 0.0
            
            # Sort by first objective
            valid.sort(key=lambda x: x[0])
            
            # Use recursive 3D hypervolume computation
            def hv_3d(pts, ref):
                if not pts:
                    return 0.0
                n = len(pts)
                if n == 1:
                    p = pts[0]
                    return max(0, ref[0] - p[0]) * max(0, ref[1] - p[1]) * max(0, ref[2] - p[2])
                
                # Sort by first objective
                pts_sorted = sorted(pts, key=lambda x: x[0])
                
                # Use sweep line approach for 3D
                total = 0.0
                for i in range(n):
                    p = pts_sorted[i]
                    # Volume dominated by p but not by any point with smaller first obj
                    # This is complex; use a simpler approach for small n
                    pass
                
                # For small populations, use direct computation
                # Hypervolume = union of boxes [p_i, ref]
                # For 3D, we can use the fact that HV = sum over non-dominated points
                # of the volume they uniquely contribute
                
                # Simpler: use the "layers" approach
                # Sort by first coordinate
                pts_by_first = sorted(pts, key=lambda x: x[0])
                
                # For each point, compute the volume it contributes
                # which is the box [p, ref] minus the union of boxes from points
                # that dominate it in at least one coordinate and are "before" it
                # This is getting complex; let's use a different approach
                
                # Use the standard 3D HV algorithm:
                # 1. Sort points by first objective
                # 2. For each point, compute the 2D HV of the remaining points
                #    in the slice where x > p[0]
                
                # Actually, for n <= 30, we can use a simpler O(n^3) approach
                # Hypervolume in 3D can be computed by:
                # HV = sum_i (x_i - x_{i-1}) * HV_2D(points with x_j > x_{i-1}, y,z)
                
                # Let's use a cleaner recursive approach
                def hv_2d(pts2d, ref2d):
                    if not pts2d:
                        return 0.0
                    pts2d = sorted(pts2d, key=lambda x: x[0])
                    total = 0.0
                    prev_y = ref2d[1]
                    for p in pts2d:
                        if p[1] < prev_y:
                            total += (ref2d[0] - p[0]) * (prev_y - p[1])
                            prev_y = p[1]
                    return total
                
                # 3D HV: sweep along first axis
                pts_sorted = sorted(pts, key=lambda x: x[0])
                total = 0.0
                prev_x = ref[0]
                
                for i, p in enumerate(pts_sorted):
                    # Find all points with x > p[0] (i.e., points after i in sorted order)
                    # and compute their 2D HV in (y,z) with reference (ref[1], ref[2])
                    # But we need points that are not dominated in y,z by p
                    
                    # Actually, the standard approach:
                    # HV = sum over i of (x_{i-1} - x_i) * HV_2D({points j: x_j >= x_{i-1}}, ref_2d)
                    # where x_{i-1} is the previous x value (starting from ref[0])
                    
                    # Let me use a different well-known formula:
                    # For 3D, sort by x. For each point i, the contribution is:
                    # (x_{i-1} - x_i) * (2D HV of points j with x_j > x_i in yz-plane)
                    
                    # This is still complex. Let's just use a simple but correct method
                    # for small n: enumerate all cells in the grid formed by point coordinates
                    # This is O(n^3) which is fine for n <= 30
                    
                    # Collect all unique x, y, z coordinates
                    xs = sorted(set([p[0] for p in pts] + [ref[0]]))
                    ys = sorted(set([p[1] for p in pts] + [ref[1]]))
                    zs = sorted(set([p[2] for p in pts] + [ref[2]]))
                    
                    # For each cell, check if it's dominated by at least one point
                    total = 0.0
                    for i in range(len(xs) - 1):
                        for j in range(len(ys) - 1):
                            for k in range(len(zs) - 1):
                                # Cell is [xs[i], xs[i+1]] x [ys[j], ys[j+1]] x [zs[k], zs[k+1]]
                                # It's counted if there exists a point p such that
                                # p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]
                                dominated = False
                                for p in pts:
                                    if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]:
                                        dominated = True
                                        break
                                if dominated:
                                    total += (xs[i+1] - xs[i]) * (ys[j+1] - ys[j]) * (zs[k+1] - zs[k])
                    
                    return total
                
                return hv_2d  # placeholder, will fix below
            
            # Use the grid-based approach directly
            xs = sorted(set([p[0] for p in valid] + [ref[0]]))
            ys = sorted(set([p[1] for p in valid] + [ref[1]]))
            zs = sorted(set([p[2] for p in valid] + [ref[2]]))
            
            total = 0.0
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    for k in range(len(zs) - 1):
                        dominated = False
                        for p in valid:
                            if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]:
                                dominated = True
                                break
                        if dominated:
                            total += (xs[i+1] - xs[i]) * (ys[j+1] - ys[j]) * (zs[k+1] - zs[k])
            
            return total

        # Main evolutionary loop
        generation = 0
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            generation += 1

            # Non-dominated sorting
            fronts = non_dominated_sort(F)

            # Build next generation using NSGA-II style selection
            next_indices = []
            next_used = 0

            # Add all non-dominated points (first front)
            for i in fronts[0]:
                next_indices.append(i)
                next_used += 1

            # Fill remaining slots with next fronts
            for front in fronts[1:]:
                if next_used >= pop_size:
                    break
                # Compute crowding distance for this front
                cd = crowding_distance(F, front)
                # Sort by crowding distance (descending)
                order = np.argsort(-cd)
                for idx in order:
                    if next_used >= pop_size:
                        break
                    i = front[idx]
                    next_indices.append(i)
                    next_used += 1

            # If we still need more, fill with random points
            while next_used < pop_size:
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                next_indices.append(-1)  # Mark as new random
                next_used += 1
                used += 1

            # Build new population
            new_X = []
            new_F = []
            for i in next_indices:
                if i == -1:
                    x_new = lo + (hi - lo) * rng.random(dim)
                    f_new = problem(x_new)
                    new_X.append(x_new)
                    new_F.append(f_new)
                else:
                    new_X.append(X[i])
                    new_F.append(F[i])

            X = np.array(new_X)
            F = np.array(new_F)

            # Generate offspring via differential evolution
            # Use CR (crossover rate) and F (scale factor)
            cr = 0.8
            # Adaptive scale factor: start higher, decrease over time
            progress = used / budget
            scale = 0.7 * (1.0 - 0.6 * progress)

            for _ in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break

                # Select three distinct random parents
                idxs = rng.choice(pop_size, size=3, replace=False)
                base = X[idxs[0]]
                v = base + scale * (X[idxs[1]] - X[idxs[2]])
                v = np.clip(v, lo, hi)

                # Crossover
                trial = X[rng.integers(pop_size)]
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < cr or j == j_rand:
                        trial[j] = v[j]

                f_trial = problem(trial)
                used += 1

                # Selection: keep the better of parent and trial
                # Use non-domination and crowding distance
                parent_idx = rng.integers(pop_size)
                if dominates(f_trial, F[parent_idx]):
                    X[parent_idx] = trial
                    F[parent_idx] = f_trial
                elif not dominates(F[parent_idx], f_trial):
                    # Non-dominated: use crowding distance
                    # Simple approach: keep trial if it's in a less crowded region
                    if rng.random() < 0.5:
                        X[parent_idx] = trial
                        F[parent_idx] = f_trial

        return X
# EVOLVE-BLOCK-END
