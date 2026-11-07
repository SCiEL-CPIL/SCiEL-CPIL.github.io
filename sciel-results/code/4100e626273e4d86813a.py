"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        
        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Hypervolume calculator for 3 objectives (exact)
        def compute_hv(points, ref):
            """Compute exact hypervolume for 3D points."""
            # Filter points that are dominated by ref
            valid = []
            for p in points:
                if all(p[i] < ref[i] for i in range(n_obj)):
                    valid.append(p)
            if not valid:
                return 0.0
            
            # Remove dominated points
            non_dom = []
            for p in valid:
                dominated = False
                for q in valid:
                    if q is not p and all(q[i] <= p[i] for i in range(n_obj)) and any(q[i] < p[i] for i in range(n_obj)):
                        dominated = True
                        break
                if not dominated:
                    non_dom.append(p)
            
            if not non_dom:
                return 0.0
            
            # Sort by first objective
            non_dom.sort(key=lambda p: p[0])
            
            hv = 0.0
            # For 3D, we can compute HV by iterating over points and computing the volume
            # that each point contributes.
            # A point contributes if it's on the Pareto front.
            # For point i, its contribution is the volume of the box from p[i] to ref,
            # minus the volume already covered by points that dominate it in the first dimension.
            
            # Better approach: sort by f0, then for each point, compute the 2D HV of the
            # remaining points in the (f1, f2) plane, weighted by the f0 extent.
            
            # Actually, for 3D, a simpler exact method:
            # Sort by f0. For each point i (in order), the volume it contributes is:
            # (f0[i+1] - f0[i]) * HV_2D of points j where f0[j] >= f0[i] in the (f1,f2) plane
            # But this is tricky. Let's use the standard exact HV algorithm for 3D.
            
            # Standard 3D HV: 
            # 1. Sort points by f0
            # 2. For each point i from 0 to n-1:
            #    - Consider the subproblem of points i..n-1
            #    - The contribution of point i is (min(f0[i+1], ref[0]) - f0[i]) * HV_2D(points i..n-1 in f1,f2)
            #    - But this overcounts. 
            #    
            # Let's use a different approach: the "contribution" of each point.
            # Point p_i contributes the volume of the box [p_i, ref] minus the union of boxes of 
            # other points that overlap with it. This is complex.
            
            # For 3D, we can use the following exact algorithm:
            # 1. Sort by f0 ascending
            # 2. Maintain a list of "active" points in the (f1,f2) plane
            # 3. For each point i, compute the 2D HV of the active points, then subtract the 
            #    contribution of the current point.
            
            # Actually, let's just implement a simple exact HV for 3D using the standard method:
            # HV = sum over all points of the volume of the box from the point to ref, 
            # minus the volume of overlaps.
            
            # A cleaner approach: use the "divide and conquer" or "sweep line" method.
            # For 3D, we can do:
            # 1. Sort by f0
            # 2. For each i, the contribution is (f0_next - f0[i]) * HV_2D(points from i onwards)
            #    where f0_next is min(f0[i+1], ref[0]) if i < n-1 else ref[0]
            #    But this isn't quite right either.
            
            # Let me just implement the standard exact 3D HV:
            # The hypervolume is the volume of the union of boxes [p_i, ref] for all i.
            # We can compute this by inclusion-exclusion, but that's exponential.
            
            # For 3D, there's an O(n^2 log n) algorithm:
            # 1. Sort points by f0
            # 2. Use a sweep line in f0, maintaining the 2D HV of the remaining points
            
            # Let's implement a simpler O(n^3) exact HV for 3D (n is small, ~20):
            # For each point p_i, compute the volume of the box [p_i, ref] that is NOT covered 
            # by any other box [p_j, ref].
            
            # Actually, the simplest exact method for small n:
            # 1. Get non-dominated points
            # 2. For each pair of points, compute the intersection of their boxes
            # 3. Use inclusion-exclusion (but this gets complex)
            
            # Let's use the standard "contribution" method:
            # Sort points by f0. For point i, its contribution is:
            # width_i * HV_2D of points j >= i in (f1, f2)
            # where width_i = min(f0[i+1], ref[0]) - f0[i] (with f0[n] = ref[0])
            # But we need to be careful: the 2D HV should only consider points that are 
            # "visible" from point i.
            
            # I'll implement a direct O(n^3) method:
            # For each point p_i, compute the volume of the region in [p_i, ref] 
            # that is not dominated by any other point.
            
            # A point p_i dominates a region if for all q in the region, p_i <= q.
            # The region [p_i, ref] is dominated by p_j if p_j <= p_i (in all dims).
            # But we're looking at the union of boxes, so we need the volume of the union.
            
            # Let's just use the standard algorithm:
            # HV_3D = sum_{i} (f0_{i+1} - f0_i) * HV_2D(S_i)
            # where S_i = {j : j >= i, f1_j < ref[1], f2_j < ref[2]}
            # and f0_{n} = ref[0]
            # But this is not correct because the 2D HV should be computed for the 
            # "slice" at f0_i.
            
            # OK, I'll implement a brute-force exact HV for 3D:
            # 1. Get non-dominated points (sorted by f0)
            # 2. For each point i, compute the 2D HV of the "shadow" cast by points i..n-1
            #    in the (f1, f2) plane, clipped to [f1_i, ref[1]] x [f2_i, ref[2]]
            # 3. The contribution of point i is (f0[i+1] - f0[i]) * that 2D HV
            #    where f0[n] = ref[0]
            
            # Wait, this is still not quite right. Let me think again.
            
            # The correct formula for 3D HV with points sorted by f0:
            # HV = sum_{i=0}^{n-1} (f0[i+1] - f0[i]) * HV_2D({j >= i})
            # where HV_2D is the 2D hypervolume of points {j >= i} in the (f1, f2) plane
            # with reference point (ref[1], ref[2]).
            # But this is only correct if the points are "layered" properly.
            
            # Let me just implement a simple exact HV by computing the union of boxes
            # using a grid-free method for 3D.
            
            # For 3D, the exact HV can be computed as follows:
            # 1. Sort points by f0
            # 2. For i from 0 to n-1:
            #    - Let f0_next = min(f0[i+1], ref[0]) if i < n-1 else ref[0]
            #    - Compute the 2D HV of points {j : j >= i} in (f1, f2) with ref (ref[1], ref[2])
            #    - Add (f0_next - f0[i]) * HV_2D to the total
            # But this overcounts because the 2D HV includes volume that may be covered 
            # by earlier points.
            
            # I think the correct approach is:
            # HV = sum_{i} (f0[i+1] - f0[i]) * HV_2D_visible(i)
            # where HV_2D_visible(i) is the 2D HV of the "visible" part of the 
            # (f1,f2) projection of points i..n-1.
            
            # This is getting complicated. Let me just implement a simple O(n^3) exact HV:
            # For each point p_i, compute the volume of the box [p_i, ref] minus the 
            # volume of the intersection with all other boxes.
            
            # Actually, for n <= 30 or so, I can just use a simple inclusion-exclusion 
            # or a direct computation.
            
            # Let's use the following method:
            # 1. Sort points by f0
            # 2. For each i, compute the 2D HV of points i..n-1 in (f1,f2)
            # 3. The contribution of point i is (f0[i+1] - f0[i]) * (HV_2D(i) - HV_2D(i+1))
            #    where HV_2D(i) is the 2D HV of points i..n-1
            #    and HV_2D(n) = 0
            # This should work because the "new" volume added by going from slice i+1 to i
            # is the difference in 2D HV.
            
            # Let me verify: 
            # HV_3D = integral_{f0=0}^{ref[0]} HV_2D(f0) df0
            # where HV_2D(f0) is the 2D HV of points with f0_j <= f0 in the (f1,f2) plane.
            # If we sort by f0, then for f0 in [f0[i], f0[i+1]), HV_2D(f0) = HV_2D(points 0..i)
            # So HV_3D = sum_{i} (f0[i+1] - f0[i]) * HV_2D(points 0..i)
            # But this is the volume of the union of boxes [ref - p_i, ref]... no, that's not right.
            
            # I'm confusing myself. Let me just implement a simple exact HV for 3D using 
            # the "contribution" method from the literature.
            
            # For now, let's use a simple approximation: sort by f0, and for each point,
            # compute the 2D HV of the remaining points.
            
            # Actually, let me just implement the standard exact 3D HV algorithm:
            # 1. Sort points by f0
            # 2. For i from n-1 down to 0:
            #    - Compute the 2D HV of points i..n-1 in (f1,f2)
            #    - The contribution of point i is (min(f0[i+1], ref[0]) - f0[i]) * HV_2D(i)
            #      where f0[n] = ref[0]
            # But this is still not correct.
            
            # Let me just use a simple O(n^2) 2D HV and integrate.
            
            def hv_2d(points_2d, ref_2d):
                """Compute 2D hypervolume exactly."""
                # Filter points
                pts = [p for p in points_2d if p[0] < ref_2d[0] and p[1] < ref_2d[1]]
                if not pts:
                    return 0.0
                # Remove dominated
                non_dom = []
                for p in pts:
                    dominated = False
                    for q in pts:
                        if q is not p and q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                            dominated = True
                            break
                    if not dominated:
                        non_dom.append(p)
                if not non_dom:
                    return 0.0
                # Sort by f1
                non_dom.sort(key=lambda p: p[0])
                hv = 0.0
                for i in range(len(non_dom)):
                    f1_next = non_dom[i+1][0] if i < len(non_dom)-1 else ref_2d[0]
                    width = min(f1_next, ref_2d[0]) - non_dom[i][0]
                    if width <= 0:
                        continue
                    height = ref_2d[1] - non_dom[i][1]
                    if height <= 0:
                        continue
                    hv += width * height
                return hv
            
            # For 3D, use the following:
            # Sort by f0
            # HV = sum_{i} (f0[i+1] - f0[i]) * HV_2D(points i..n-1 in f1,f2)
            # where f0[n] = ref[0]
            # But this overcounts. The correct formula is:
            # HV = sum_{i} (f0[i+1] - f0[i]) * HV_2D_visible(i)
            # where HV_2D_visible(i) is the 2D HV of the "new" volume added by point i.
            
            # Let me just use a simple approach: compute the 2D HV for each "layer" 
            # and take the difference.
            
            # Sort by f0
            non_dom.sort(key=lambda p: p[0])
            n = len(non_dom)
            
            # Compute cumulative 2D HV
            cum_hv_2d = [0.0] * (n + 1)
            for i in range(n, 0, -1):
                pts_2d = [(non_dom[j][1], non_dom[j][2]) for j in range(i-1, n)]
                cum_hv_2d[i-1] = hv_2d(pts_2d, [ref[1], ref[2]])
            cum_hv_2d[n] = 0.0
            
            # Now, the contribution of point i is (f0[i+1] - f0[i]) * (cum_hv_2d[i] - cum_hv_2d[i+1])
            # Wait, that's not right either.
            
            # Let me think differently. The 3D HV is the volume of the union of boxes.
            # I'll just compute it directly by summing the contributions.
            
            # For point i (sorted by f0), its "unique" contribution is the volume of the 
            # box [p_i, ref] that is not covered by any other box.
            # This is hard to compute exactly.
            
            # For now, let's use a simple approximation that works well in practice:
            # Sort by f0, and for each point, compute the 2D HV of the "shadow" 
            # of the remaining points.
            
            # Actually, let's just use the formula:
            # HV = sum_{i=0}^{n-1} (f0[i+1] - f0[i]) * HV_2D(points 0..i in f1,f2)
            # where f0[n] = ref[0]
            # This computes the volume under the "upper envelope" of the 2D HV.
            
            # Let me test this with a simple case:
            # Points: (1,1,1), (2,2,2), ref: (3,3,3)
            # Sorted by f0: (1,1,1), (2,2,2)
            # HV_2D({(1,1)}) = (3-1)*(3-1) = 4
            # HV_2D({(1,1),(2,2)}) = (3-1)*(3-1) + (3-2)*(3-2) = 4 + 1 = 5? 
            # No, the 2D HV of {(1,1),(2,2)} with ref (3,3) is:
            #   Point (1,1): box [1,3]x[1,3], area = 4
            #   Point (2,2): box [2,3]x[2,3], area = 1
            #   Union area = 4 (since (2,2) is dominated by (1,1) in 2D)
            # So HV_2D = 4
            # HV_3D = (2-1)*4 + (3-2)*4 = 4 + 4 = 8
            # Actual HV: box [1,3]^3 has volume 8, but we need to subtract the part 
            # dominated by (2,2,2)... no, (2,2,2) is inside the box [1,3]^3.
            # The union of boxes [1,3]^3 and [2,3]^3 is just [1,3]^3, so HV = 8. Correct!
            
            # Let me try another case:
            # Points: (1,3,1), (3,1,1), ref: (4,4,4)
            # Sorted by f0: (1,3,1), (3,1,1)
            # HV_2D({(3,1)}) = (4-3)*(4-1) = 3
            # HV_2D({(3,1),(1,1)}) = (4-3)*(4-1) + (4-1)*(4-1) = 3 + 9 = 12? 
            # No, the 2D HV of {(3,1),(1,1)} with ref (4,4):
            #   Point (1,1): box [1,4]x[1,4], area = 9
            #   Point (3,1): box [3,4]x[1,4], area = 3
            #   Union area = 9 (since (3,1) is dominated by (1,1) in 2D)
            # So HV_2D = 9
            # HV_3D = (3-1)*3 + (4-3)*9 = 6 + 9 = 15
            # Actual HV: 
            #   Box 1: [1,4]x[3,4]x[1,4], volume = 3*1*3 = 9
            #   Box 2: [3,4]x[1,4]x[1,4], volume = 1*3*3 = 9
            #   Intersection: [3,4]x[3,4]x[1,4], volume = 1*1*3 = 3
            #   Union volume = 9 + 9 - 3 = 15. Correct!
            
            # Great, the formula works!
            
            # So: HV_3D = sum_{i=0}^{n-1} (f0[i+1] - f0[i]) * HV_2D(points 0..i in f1,f2)
            # where f0[n] = ref[0]
            
            total_hv = 0.0
            for i in range(n):
                f0_next = non_dom[i+1][0] if i < n-1 else ref[0]
                width = min(f0_next, ref[0]) - non_dom[i][0]
                if width <= 0:
                    continue
                pts_2d = [(non_dom[j][1], non_dom[j][2]) for j in range(i+1)]
                hv_2d_val = hv_2d(pts_2d, [ref[1], ref[2]])
                total_hv += width * hv_2d_val
            
            return total_hv
        
        # Initialize population
        N = 20
        if N > self.budget:
            N = self.budget
            
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            front = [set() for _ in range(n)]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_doms[p] += 1
                if n_doms[p] == 0:
                    ranks[p] = 0
                    front[0].add(p)
                    
            i = 0
            while front[i]:
                next_front = set()
                for p in front[i]:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            ranks[q] = i + 1
                            next_front.add(q)
                i += 1
                if next_front:
                    front[i] = next_front
                else:
                    break
            return [list(f) for f in front if f]

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            m = len(F)
            n_obj = self.n_obj
            crowd = np.zeros(m)
            
            for k in range(n_obj):
                sorted_idx = sorted(front_indices, key=lambda i: F[i][k])
                crowd[sorted_idx[0]] = float('inf')
                crowd[sorted_idx[-1]] = float('inf')
                f_min = F[sorted_idx[0]][k]
                f_max = F[sorted_idx[-1]][k]
                if f_max == f_min:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    crowd[idx] += (F[sorted_idx[i+1]][k] - F[sorted_idx[i-1]][k]) / (f_max - f_min)
            
            return {i: crowd[i] for i in front_indices}

        # Helper: Binary Tournament
        def tournament_select(pop_F, ranks, crowds):
            i = np.random.randint(len(pop_F))
            j = np.random.randint(len(pop_F))
            if ranks[i] < ranks[j]:
                return i
            elif ranks[j] < ranks[i]:
                return j
            else:
                if crowds[i] > crowds[j]:
                    return i
                else:
                    return j

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutate(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                x_mut[i] += delta_q * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Sort current population
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # Calculate crowding distance for all fronts
            crowds = np.zeros(len(F))
            for front in fronts:
                cd_dict = crowding_distance(F, front)
                for idx, c in cd_dict.items():
                    crowds[idx] = c

            # Generate offspring
            Q_X = []
            Q_F = []
            
            # We can generate up to min(N, self.budget - used) offspring
            num_offspring = min(N, self.budget - used)
            
            for _ in range(num_offspring):
                # Selection
                p1_idx = tournament_select(F, ranks, crowds)
                p2_idx = tournament_select(F, ranks, crowds)
                
                # Crossover
                child1, child2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                child1 = polynomial_mutate(child1)
                child2 = polynomial_mutate(child2)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                
                Q_X.append(child1)
                Q_F.append(f1)
                used += 1
                if used >= self.budget:
                    break
                
                Q_X.append(child2)
                Q_F.append(f2)
                used += 1
                if used >= self.budget:
                    break
            
            if not Q_X:
                break

            Q_X = np.array(Q_X)
            Q_F = np.array(Q_F)
            
            # Combine P and Q
            R_X = np.vstack((X, Q_X))
            R_F = np.vstack((F, Q_F))
            
            # Sort R
            fronts_R = fast_non_dominated_sort(R_F)
            
            # Select N individuals for next generation
            next_X = []
            next_F = []
            next_ranks = np.zeros(len(R_F), dtype=int)
            for i, front in enumerate(fronts_R):
                for idx in front:
                    next_ranks[idx] = i
            
            # We need to select N individuals from R
            # Iterate through fronts, adding entire fronts until we exceed N
            # Then fill the rest from the next front based on crowding distance
            selected_indices = []
            for front in fronts_R:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Need to pick only some from this front
                    # Calculate crowding distance for this front
                    cd_dict = crowding_distance(R_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    needed = N - len(selected_indices)
                    selected_indices.extend(sorted_front[:needed])
                    break
            
            X = R_X[selected_indices]
            F = R_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
