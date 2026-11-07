"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer for RE34 (5 vars, 3 obj)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a good chunk of budget for initial exploration
        pop_size = min(60, budget // 3)
        
        # Initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (from task)
        ref = np.array([1705.0, 12.0, 0.3])

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sorting. Returns list of fronts."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # domination count
            n_dom = [0] * n  # number of solutions dominated by i
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            dist = {i: 0.0 for i in indices}
            for m in range(n_obj):
                sorted_idx = sorted(indices, key=lambda i: F_arr[i, m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_min = F_arr[sorted_idx[0], m]
                f_max = F_arr[sorted_idx[-1], m]
                if f_max == f_min:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[k]] += (F_arr[sorted_idx[k + 1], m] - F_arr[sorted_idx[k - 1], m]) / (f_max - f_min)
            return dist

        def hypervolume_3d(points, ref_point):
            """Compute hypervolume for 3D points strictly dominated by ref_point."""
            # Filter points that are strictly better than ref in all objectives
            valid = [p for p in points if np.all(p < ref_point)]
            if not valid:
                return 0.0
            
            # Sort by first objective
            valid.sort(key=lambda p: p[0])
            
            # For each point, compute its contribution
            # A point contributes if it's non-dominated
            # Use a simple approach: for each point, find the volume it uniquely covers
            
            # More efficient: sort by obj1, then for consecutive points, 
            # the contribution of point i is (obj1[i+1] - obj1[i]) * (ref[1] - max(obj2 of points with obj1 < obj1[i] that dominate in obj2,3))...
            # Actually, let's use a simpler but correct approach for small n
            
            n_pts = len(valid)
            if n_pts == 1:
                p = valid[0]
                return (ref_point[0] - p[0]) * (ref_point[1] - p[1]) * (ref_point[2] - p[2])
            
            # Sort by first objective
            valid.sort(key=lambda p: p[0])
            
            hv = 0.0
            # For each point, compute the volume of the box it uniquely dominates
            # that is not dominated by any other point
            for i in range(n_pts):
                p = valid[i]
                # The region uniquely dominated by p in obj1 is [p[0], next_obj1)
                if i < n_pts - 1:
                    next_obj1 = valid[i + 1][0]
                else:
                    next_obj1 = ref_point[0]
                
                # In this obj1 slice, we need to find the max obj2 and obj3
                # that are not "covered" by other points with obj1 <= p[0]
                # Actually, this is complex. Let's use a different approach.
                
            # Simpler approach: use the inclusion-exclusion or just compute for small n
            # Since n_pts is at most ~60, we can use a brute-force-ish approach
            
            # Actually, let's use a proper 3D hypervolume algorithm
            # Sort by x, then maintain a 2D structure for y,z
            
            # For simplicity and correctness with small n, let's use:
            # hv = sum over all subsets... no, that's 2^n.
            
            # Let's use the standard approach:
            # Sort points by x coordinate
            # For each point, the contribution is (x_next - x_i) * area_in_2d(y,z)
            # where area_in_2d is the 2D hypervolume of the points with x <= x_i 
            # restricted to the region beyond x_i
            
            # Actually, the standard 3D HV algorithm:
            # Sort by x. Process from right to left (largest x first).
            # Maintain a list of "active" points in 2D (y,z).
            # The contribution of point i is (x_{i-1} - x_i) * HV_2D(active_points)
            # where active_points are those with x < x_i that are non-dominated in (y,z)
            
            # Let me implement a cleaner version:
            
            # Filter to non-dominated points first
            non_dom = []
            for p in valid:
                is_dom = False
                for q in valid:
                    if q is p:
                        continue
                    if np.all(q <= p) and np.any(q < p):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom.append(p)
            
            if not non_dom:
                return 0.0
            
            # Sort by first objective
            non_dom.sort(key=lambda p: p[0])
            n_nd = len(non_dom)
            
            hv = 0.0
            for i in range(n_nd):
                p = non_dom[i]
                x_next = non_dom[i + 1][0] if i < n_nd - 1 else ref_point[0]
                
                # For the slice [p[0], x_next), the 2D region is determined by
                # the points with index <= i (i.e., obj1 <= p[0])
                # We need the 2D hypervolume of the set of (obj2, obj3) values
                # from points 0..i, clipped to [0, ref[1]] x [0, ref[2]]
                
                # Collect (y,z) for points 0..i
                yz = [(non_dom[j][1], non_dom[j][2]) for j in range(i + 1)]
                
                # Compute 2D hypervolume
                # Sort by y
                yz.sort(key=lambda t: t[0])
                area = 0.0
                max_z = 0.0
                prev_y = 0.0
                # Actually, 2D HV: sort by y, keep track of max z
                # The area is sum of (y_next - y_curr) * min(max_z, ref_z) ... no
                
                # 2D hypervolume: 
                # Sort points by y. The HV is the area under the "skyline"
                # in the (y,z) plane, bounded by ref.
                
                # For each consecutive pair in sorted-by-y, the width is dy,
                # and the height is the minimum z among all points with y' <= y_i
                # that are on the Pareto front in 2D.
                
                # Let me just compute it directly:
                # For 2D, HV = sum over non-dominated points of (y_{i+1} - y_i) * z_i
                # where points are sorted by y and we take the Pareto front in 2D.
                
                # Find 2D non-dominated points
                nd2d = []
                for pt in yz:
                    is_dom2d = False
                    for q in yz:
                        if q == pt:
                            continue
                        if q[0] <= pt[0] and q[1] <= pt[1] and (q[0] < pt[0] or q[1] < pt[1]):
                            is_dom2d = True
                            break
                    if not is_dom2d:
                        nd2d.append(pt)
                
                if not nd2d:
                    continue
                    
                nd2d.sort(key=lambda t: t[0])
                n_2d = len(nd2d)
                
                for j in range(n_2d):
                    y_curr = nd2d[j][0]
                    y_next = nd2d[j + 1][0] if j < n_2d - 1 else ref_point[1]
                    # The height at this y-slice is nd2d[j][1], but capped by ref
                    # And also, we need to ensure this point is the one that sets the boundary
                    # In 2D, after sorting by y, the Pareto front has decreasing z
                    # So the contribution is (y_next - y_curr) * min(nd2d[j][1], ref_point[2])
                    # But wait, we also need to clip y to ref
                    y_curr_clipped = min(y_curr, ref_point[1])
                    y_next_clipped = min(y_next, ref_point[1])
                    if y_next_clipped > y_curr_clipped:
                        area += (y_next_clipped - y_curr_clipped) * min(nd2d[j][1], ref_point[2])
                
                hv += (x_next - p[0]) * area
            
            return hv

        def select_parents(F_arr, pop_size):
            """Select parents using NSGA-II style tournament."""
            fronts = fast_non_dominated_sort(F_arr)
            
            # Assign rank
            rank = np.zeros(len(F_arr), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Crowding distance per front
            cd = np.zeros(len(F_arr))
            for front in fronts:
                if front:
                    cd_dict = crowding_distance(F_arr, front)
                    for idx in front:
                        cd[idx] = cd_dict[idx]
            
            # Tournament selection
            parents = []
            while len(parents) < pop_size:
                i = rng.integers(len(F_arr))
                j = rng.integers(len(F_arr))
                if rank[i] < rank[j]:
                    parents.append(i)
                elif rank[j] < rank[i]:
                    parents.append(j)
                else:
                    if cd[i] > cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        # Main loop
        while used < budget:
            if time.time() > deadline - 5:
                break
            
            # Create offspring
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break
            
            parents = select_parents(F, pop_size)
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                p1 = parents[rng.integers(len(parents))]
                p2 = parents[rng.integers(len(parents))]
                
                # SBX crossover
                if rng.random() < 0.9:
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 20))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 20))
                            child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 20))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 20))
                            child[d] = 0.5 * ((1 + beta) * X[p2, d] + (1 - beta) * X[p1, d])
                else:
                    child = X[p1].copy()
                
                # Mutation (polynomial)
                for d in range(dim):
                    if rng.random() < 0.3:
                        u = rng.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1 / (2 + 20)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (2 + 20))
                        child[d] += delta * (hi[d] - lo[d])
                
                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline - 2:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Select next generation
            fronts = fast_non_dominated_sort(F_comb)
            
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd_dict = crowding_distance(F_comb, front)
                    front_sorted = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    for idx in front_sorted[:remaining]:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                    break
            
            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)
        
        return X
# EVOLVE-BLOCK-END
