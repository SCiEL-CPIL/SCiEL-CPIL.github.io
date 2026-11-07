import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 295.0  # 5s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # With budget 200, a pop size of 20 allows ~10 generations if we generate 1 child per iter
        # Or we can generate a full new population each generation?
        # Standard NSGA-II: Pop N, generate N offspring, select N from 2N.
        # Budget 200. If N=20, we can do 200/20 = 10 generations.
        # This is reasonable.
        pop_size = 20
        
        # Initialize
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def hypervolume_3d(points, ref_point):
            """
            Calculate the 3D hypervolume of a set of points dominated by ref_point.
            points: (N, 3) array
            ref_point: (3,) array
            Returns: float
            """
            # Filter points that are strictly dominated by ref_point (i.e., better than ref in all objectives)
            # Since we are minimizing, we want points where f_i < ref_i for all i.
            mask = np.all(points < ref_point, axis=1)
            pts = points[mask]
            
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            # We calculate the volume contribution of each point.
            # For 3D, a common efficient method is O(N log N) or O(N^2).
            # Given N <= 40, O(N^2) is perfectly fine.
            
            # We can compute the hypervolume by summing the volume of boxes
            # where each point is the "top-right" corner (in minimization, bottom-left).
            # However, overlapping boxes must be handled.
            
            # Alternative: Recursive decomposition or Sweep Line.
            # For small N, a simple recursive approach or even brute-force inclusion-exclusion is risky.
            # Let's use a standard O(N^2) or O(N log N) algorithm for 3D.
            
            # A robust simple method for small N:
            # Sort by x. Iterate. For each point, calculate the area in y-z plane
            # that is not covered by previous points (which have smaller x).
            
            n = len(pts)
            hv = 0.0
            
            # pts is sorted by x (obj 1)
            for i in range(n):
                x_i = pts[i, 0]
                y_i = pts[i, 1]
                z_i = pts[i, 2]
                
                # The x-width for this point is determined by the next point's x or ref_point[0]
                if i == n - 1:
                    x_next = ref_point[0]
                else:
                    x_next = pts[i+1, 0]
                
                width_x = x_next - x_i
                if width_x <= 0:
                    continue
                
                # Now we need the 2D hypervolume in the y-z plane
                # for the points j >= i, projected onto y-z, with reference (ref_y, ref_z)
                # But wait, the "active" points for the y-z plane at this x-slice are
                # those points j such that x_j >= x_i. Since we are iterating from smallest x,
                # all points from i to n-1 are "active" in the sense that they extend to the right.
                # However, the volume contributed by point i is width_x * Area_2D(active_points)
                # where Area_2D is the hypervolume of the set of points {pts[i], pts[i+1], ...}
                # in the y-z plane, bounded by ref_y and ref_z.
                
                # Actually, the standard sweep line algorithm for 3D HV:
                # 1. Sort points by x.
                # 2. Maintain a list of "active" points in y-z space.
                # 3. The contribution of the x-interval [x_i, x_{i+1}] is (x_{i+1}-x_i) * HV_2D(active_points).
                # The active points for the interval [x_i, x_{i+1}] are all points with x >= x_i.
                # So for interval i, active set is pts[i:].
                
                # Let's compute HV_2D for pts[i:] in y-z plane.
                # pts[i:] has size n-i.
                # We can do this in O(K^2) for each i, leading to O(N^3) total.
                # N=40, N^3 = 64000 operations. This is very fast.
                
                # Let's optimize slightly:
                # We can just compute HV_2D for the subarray pts[i:].
                
                def hv_2d(sub_pts, ref_y, ref_z):
                    # sub_pts: (K, 3) but we only care about cols 1, 2
                    # Filter points better than ref in y and z
                    mask_2d = np.all(sub_pts[:, 1:3] < [ref_y, ref_z], axis=1)
                    pts_2d = sub_pts[mask_2d][:, 1:3]
                    
                    if len(pts_2d) == 0:
                        return 0.0
                    
                    # Sort by y
                    pts_2d = pts_2d[np.argsort(pts_2d[:, 0])]
                    k = len(pts_2d)
                    area = 0.0
                    
                    for j in range(k):
                        y_j = pts_2d[j, 0]
                        z_j = pts_2d[j, 1]
                        
                        if j == k - 1:
                            y_next = ref_y
                        else:
                            y_next = pts_2d[j+1, 0]
                        
                        width_y = y_next - y_j
                        if width_y <= 0:
                            continue
                        
                        # Height in z is determined by the minimum z among points j..k-1
                        # Wait, for 2D HV (minimization), the area is sum of (y_next - y_j) * (ref_z - min_z)
                        # where min_z is the min z of points from j to k-1?
                        # No, the "skyline" is the lower envelope.
                        # If we sort by y, the contribution of point j is (y_next - y_j) * (ref_z - z_j)
                        # ONLY IF z_j is smaller than all z's to its right?
                        # No, that's for max.
                        # For min, the hypervolume is the area under the curve formed by the points,
                        # bounded by the axes (ref).
                        # If we sort by y ascending, the "active" height for the interval [y_j, y_next]
                        # is determined by the point with the smallest z in the set {j, ..., k-1}?
                        # Actually, yes. The hypervolume is the union of boxes.
                        # In 2D, if we sort by y, the contribution of the interval [y_j, y_{j+1}]
                        # is (y_{j+1} - y_j) * (ref_z - min_{k>=j}(z_k)).
                        
                        min_z = np.min(pts_2d[j:, 1])
                        height_z = ref_z - min_z
                        if height_z < 0:
                            height_z = 0
                        area += width_y * height_z
                    
                    return area
                
                area_2d = hv_2d(pts[i:], ref_point[1], ref_point[2])
                hv += width_x * area_2d
                
            return hv

        def select_best_hv(X_pool, F_pool, pop_size, ref_point):
            """
            Select pop_size individuals from X_pool/F_pool that maximize the 3D hypervolume.
            This is a greedy approximation or exact search for small sizes?
            Exact search for subset of size K from N is combinatorial.
            However, a common heuristic is to sort by non-dominated rank and then HV,
            or simply use a greedy approach: add points that increase HV the most.
            
            Given the small size (N=40, K=20), we can try a greedy approach:
            1. Start with empty set.
            2. Repeatedly add the point that gives the largest increase in HV.
            3. Stop when K points are selected.
            
            This is O(K * N * Cost_HV). Cost_HV is O(N^3) or similar.
            20 * 40 * 40^3 is too slow? 20*40*64000 = 51.2M operations. Might be too slow in Python.
            
            Alternative: Use NSGA-II non-dominated sorting first to filter candidates,
            then use HV on the first few fronts.
            
            Let's stick to a simpler, faster selection:
            1. Non-dominated sort.
            2. Take all points from Front 0.
            3. If Front 0 > pop_size, select best by HV among Front 0.
               (Greedy HV selection on Front 0 might be acceptable if Front 0 is small, e.g., < 30).
            4. If Front 0 < pop_size, take all Front 0, then take from Front 1, etc.
            
            Let's implement a hybrid:
            - Compute non-dominated fronts.
            - Fill population with fronts.
            - If the last front needs truncation, use HV to select.
            """
            n = len(F_pool)
            # Non-dominated sort
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_pool[i] <= F_pool[j]) and np.any(F_pool[i] < F_pool[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_pool[j] <= F_pool[i]) and np.any(F_pool[j] < F_pool[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(selected)
                    # Select best 'remaining' points from 'front' based on HV
                    # Greedy HV selection
                    candidates = list(front)
                    chosen = []
                    
                    # To speed up, we can pre-filter or just do greedy if front is small
                    # Fronts in NSGA-II are usually small.
                    while len(chosen) < remaining:
                        best_idx = -1
                        best_hv = -1
                        current_hv = hypervolume_3d(F_pool[np.array(chosen)] if chosen else np.empty((0,3)), ref_point)
                        
                        for idx in candidates:
                            if idx in chosen:
                                continue
                            # Test adding idx
                            test_set = chosen + [idx]
                            test_hv = hypervolume_3d(F_pool[np.array(test_set)], ref_point)
                            if test_hv > best_hv:
                                best_hv = test_hv
                                best_idx = idx
                        
                        if best_idx == -1:
                            break
                        chosen.append(best_idx)
                    
                    selected.extend(chosen)
                    break
            
            return selected

        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Select 2 parents for DE mutation
            r1 = np.random.randint(0, pop_size)
            r2 = np.random.randint(0, pop_size)
            r3 = np.random.randint(0, pop_size)
            
            # DE/CR mutation with crossover
            # v = x[r1] + F * (x[r2] - x[r3])
            F_mut = 0.8
            CR = 0.5
            v = X[r1].copy()
            j_rand = np.random.randint(0, n_dim)
            for j in range(n_dim):
                if np.random.random() < CR or j == j_rand:
                    v[j] = X[r1][j] + F_mut * (X[r2][j] - X[r3][j])
            
            v = np.clip(v, lo, hi)
            
            fv = problem(v)
            used += 1
            
            # Combine
            X_new = np.vstack([X, v])
            F_new = np.vstack([F, fv])
            
            # Selection using Hypervolume
            selected_indices = select_best_hv(X_new, F_new, pop_size, ref_point)
            
            X = X_new[selected_indices]
            F = F_new[selected_indices]
            
        return X
