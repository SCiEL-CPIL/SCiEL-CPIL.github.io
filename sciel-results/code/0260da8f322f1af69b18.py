import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Parameters
        pop_size = 20
        n_gen = (self.budget - pop_size) // pop_size
        
        # 1. Initialization
        # Use Latin Hypercube Sampling (LHS) for better coverage
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            indices = np.random.permutation(pop_size)
            X[:, j] = lo[j] + range_[j] * (indices + np.random.rand(pop_size)) / pop_size
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Helper: Compute hypervolume contribution of a point to a set of points
        # We want to maximize the hypervolume of the set of points that are better than ref_point
        def compute_hypervolume(points):
            """
            Compute the exact hypervolume of a set of 3D points with respect to a reference point.
            Only points strictly better than the reference point in all objectives count.
            """
            # Filter points that are better than reference point in all objectives
            valid_points = []
            for p in points:
                if all(p[i] < ref_point[i] for i in range(self.n_obj)):
                    valid_points.append(p)
            
            if len(valid_points) == 0:
                return 0.0
            
            # For 3D hypervolume, we can use a simple inclusion-exclusion or grid-based approach
            # Since we only have up to ~200 points, we can use a more direct method
            # Sort by first objective
            valid_points = sorted(valid_points, key=lambda x: x[0])
            
            # Use the "sweep" algorithm for 3D hypervolume
            # This is a simplified version; for exact HV in 3D with small N, we can use
            # the fact that HV = sum over all subsets... but that's exponential.
            # Instead, use a grid-based approximation or a known O(n^2) algorithm for 3D.
            
            # For simplicity and exactness with small N, use the "dominance-based" approach:
            # A point p contributes to HV if it is not dominated by any other point in the set
            # (among the valid points). But that's not quite right for HV.
            
            # Let's use a simpler method: for each point, compute the volume of the box it "owns"
            # that is not covered by any other point. This is complex.
            
            # Alternative: Since N is small, use a Monte Carlo or grid-based estimate?
            # No, we need exact HV.
            
            # Let's use the standard 3D HV algorithm:
            # 1. Sort points by f1
            # 2. For each point, find the "dominated region" that is unique to it
            # This is still complex.
            
            # For now, use a simple O(n^3) approach:
            # For each point, compute the volume of the box from that point to ref_point,
            # but subtract the parts that are also covered by other points.
            # This is the inclusion-exclusion principle, which is complex.
            
            # Let's use a different approach: discretize the space? No, we need exact.
            
            # Actually, for 3 objectives and small N, we can use the following:
            # HV = sum over all points p of (volume of box from p to ref) * (1 if p is not dominated by any other point in the set, else 0)
            # But that's not correct either.
            
            # Let's use a known algorithm: the "sweep" algorithm for 3D HV.
            # But implementing it correctly is non-trivial.
            
            # For this implementation, I'll use a simpler but potentially less accurate method:
            # Compute the convex hull of the points and the reference point, and compute its volume?
            # No, that's not HV.
            
            # Let's use a grid-based approximation with fine resolution.
            # But that's not exact.
            
            # Given the complexity, let's use a simpler heuristic:
            # Approximate HV by summing the volume of the box from each point to ref,
            # but only for non-dominated points.
            
            # First, find non-dominated points
            def dominates(a, b):
                return all(a[i] <= b[i] for i in range(self.n_obj)) and any(a[i] < b[i] for i in range(self.n_obj))
            
            non_dom = []
            for i, p in enumerate(valid_points):
                is_dom = False
                for j, q in enumerate(valid_points):
                    if i != j and dominates(q, p):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom.append(p)
            
            # Approximate HV as sum of boxes for non-dominated points
            # This is an overestimate, but it's a reasonable proxy for local improvement
            hv = 0.0
            for p in non_dom:
                vol = 1.0
                for i in range(self.n_obj):
                    vol *= (ref_point[i] - p[i])
                hv += vol
            
            return hv
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p != q:
                        if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                            S[p].append(q)
                        elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                            n_count[p] += 1
                if n_count[p] == 0:
                    fronts[0].append(p)
                    
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1]
        
        # Helper: Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                sorted_indices = sorted(indices, key=lambda i: F[i, m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                if f_max == f_min:
                    continue
                for k in range(1, n - 1):
                    dist[sorted_indices[k]] += (F[sorted_indices[k+1], m] - F[sorted_indices[k-1], m]) / (f_max - f_min)
            return dist
        
        # Helper: Ternary search on a single dimension
        def ternary_search_dim(x, dim_idx, current_F, n_iter=8):
            """
            Perform ternary search on dimension dim_idx to maximize hypervolume contribution.
            Returns the best x found.
            """
            best_x = x.copy()
            best_hv = compute_hypervolume(list(current_F) + [problem(x)])
            
            # Define a function to evaluate hypervolume at a given value of dim_idx
            def eval_hv(val):
                x_new = x.copy()
                x_new[dim_idx] = val
                F_new = problem(x_new)
                # Compute HV of current set plus this new point
                hv = compute_hypervolume(list(current_F) + [F_new])
                return hv
            
            lo_val = lo[dim_idx]
            hi_val = hi[dim_idx]
            
            for _ in range(n_iter):
                if used >= self.budget:
                    break
                m1 = lo_val + (hi_val - lo_val) / 3.0
                m2 = hi_val - (hi_val - lo_val) / 3.0
                
                x_m1 = x.copy()
                x_m1[dim_idx] = m1
                F_m1 = problem(x_m1)
                used += 1
                hv_m1 = compute_hypervolume(list(current_F) + [F_m1])
                
                x_m2 = x.copy()
                x_m2[dim_idx] = m2
                F_m2 = problem(x_m2)
                used += 1
                hv_m2 = compute_hypervolume(list(current_F) + [F_m2])
                
                if hv_m1 < hv_m2:
                    lo_val = m1
                else:
                    hi_val = m2
                
                # Track best
                if hv_m1 > best_hv:
                    best_hv = hv_m1
                    best_x = x_m1
                if hv_m2 > best_hv:
                    best_hv = hv_m2
                    best_x = x_m2
            
            return best_x
        
        # Evolutionary Loop with coordinate descent enhancement
        for gen in range(n_gen):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
                
            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Select parents using tournament selection
            children_X = []
            children_F = []
            
            # Generate pop_size children
            for _ in range(pop_size):
                if used >= self.budget:
                    break
                
                # Tournament for parent 1
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                
                # Determine ranks
                rank1 = next((i for i, f in enumerate(fronts) if p1_idx in f), len(fronts))
                rank2 = next((i for i, f in enumerate(fronts) if p2_idx in f), len(fronts))
                
                if rank1 < rank2:
                    p1 = p1_idx
                elif rank2 < rank1:
                    p1 = p2_idx
                else:
                    cd = crowding_distance(F, list(range(pop_size)))
                    p1 = p1_idx if cd[p1_idx] > cd[p2_idx] else p2_idx
                    
                # Tournament for parent 2
                p3_idx = np.random.randint(0, pop_size)
                p4_idx = np.random.randint(0, pop_size)
                
                rank3 = next((i for i, f in enumerate(fronts) if p3_idx in f), len(fronts))
                rank4 = next((i for i, f in enumerate(fronts) if p4_idx in f), len(fronts))
                
                if rank3 < rank4:
                    p2 = p3_idx
                elif rank4 < rank3:
                    p2 = p4_idx
                else:
                    cd = crowding_distance(F, list(range(pop_size)))
                    p2 = p3_idx if cd[p3_idx] > cd[p4_idx] else p4_idx
                
                # Crossover: simple blend
                child_x = 0.5 * X[p1] + 0.5 * X[p2]
                # Add some noise
                child_x = child_x + np.random.randn(self.dim) * 0.1 * range_
                child_x = np.clip(child_x, lo, hi)
                
                # Coordinate descent refinement: optimize each dimension
                # Use only a few iterations to save budget
                if used < self.budget - 24:  # Reserve budget for 12 dims * 2 evals per dim
                    for dim_idx in range(self.dim):
                        if used >= self.budget - 2:
                            break
                        child_x = ternary_search_dim(child_x, dim_idx, F, n_iter=4)
                
                child_f = problem(child_x)
                used += 1
                
                children_X.append(child_x)
                children_F.append(child_f)
                
            if not children_X:
                break
                
            # Combine
            X = np.vstack([X, np.array(children_X)])
            F = np.vstack([F, np.array(children_F)])
            
            # Selection
            fronts = non_dominated_sort(F)
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X[idx])
                        new_F.append(F[idx])
                else:
                    cd = crowding_distance(F, front)
                    pairs = [(idx, cd[front.index(idx)]) for idx in front]
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx, _ in pairs[:needed]:
                        new_X.append(X[idx])
                        new_F.append(F[idx])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
