import numpy as np
import time

class Optimizer:
    """
    Direct Hierarchical Multi-Objective Optimizer with Adaptive Local Search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_X.append(x)
            all_F.append(f)
            eval_count += 1
            return f

        # Helper: Check if point A dominates point B (minimization)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Get non-dominated indices from current lists
        def get_non_dominated_indices():
            if not all_F:
                return []
            F = np.array(all_F)
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not nd_mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Helper: Calculate Hypervolume Contribution in 2D
        # Points must be non-dominated and strictly better than ref
        def hv_contribution_2d(point, nd_points, ref):
            # nd_points: list of tuples (f1, f2)
            # ref: tuple (r1, r2)
            # We assume nd_points are sorted by f1 ascending
            # Contribution is the area of the rectangle defined by point and the "corner"
            # formed by the next point with higher f1 and lower f2 (or ref).
            
            p1, p2 = point
            r1, r2 = ref
            
            # If point is not better than ref, contribution is 0
            if p1 >= r1 or p2 >= r2:
                return 0.0
            
            # Find the next point in the front with f1 > p1
            # Since we are in 2D and non-dominated, if f1 increases, f2 must decrease.
            # We iterate through the sorted list.
            # The "width" of the contribution is determined by the next point's f1 (or ref f1 if none).
            # The "height" is determined by the current point's f2 (or ref f2 if it's the last).
            
            # Actually, simpler logic for 2D HV contribution of a single point to the set:
            # The contribution is (next_f1 - current_f1) * (current_f2 - ref_f2) ? No.
            # Let's use the standard definition.
            # Hypervolume of a set is the volume of the union of boxes [f_i, ref].
            # In 2D, if sorted by f1 ascending (f2 descending):
            # HV = sum_{i} (f_{i+1,1} - f_{i,1}) * (ref_2 - f_{i,2}) where f_{n+1,1} = ref_1.
            # Wait, that's if we sum over the "steps".
            # The contribution of point i is the part of the HV that disappears if we remove point i.
            # If we remove point i, the box [f_i, ref] loses the area that is not covered by other boxes.
            # In 2D, if points are sorted by f1 asc:
            # The area contributed by point i is (f_{i+1,1} - f_{i,1}) * (f_{i,2} - ref_2) ? No.
            # Let's visualize.
            # Points: A(1, 4), B(2, 3), C(3, 2). Ref(5, 5).
            # Box A: [1,5]x[4,5] -> Area 4*1=4. But part is covered by B?
            # B: [2,5]x[3,5] -> Area 3*2=6.
            # C: [3,5]x[2,5] -> Area 2*3=6.
            # Union:
            # x in [1,2]: y in [4,5] -> 1*1 = 1
            # x in [2,3]: y in [3,5] -> 1*2 = 2
            # x in [3,5]: y in [2,5] -> 2*3 = 6
            # Total HV = 1 + 2 + 6 = 9.
            
            # Contribution of A: If removed, we lose x in [1,2] y in [4,5]. Area = 1.
            # Formula: (f_{i+1,1} - f_{i,1}) * (f_{i,2} - f_{i+1,2})? No.
            # For A (i=0): next is B. Width = 2-1=1. Height = 5-4=1? No, height is ref_2 - f_i_2 = 5-4=1.
            # So Contrib(A) = (f_{next,1} - f_i,1) * (ref_2 - f_i,2) ?
            # Let's check B. Next is C. Width = 3-2=1. Height = 5-3=2. Contrib(B) = 2.
            # Let's check C. Next is Ref. Width = 5-3=2. Height = 5-2=3. Contrib(C) = 6.
            # Sum = 1+2+6=9. Correct.
            
            # So for a point P_i at index i in sorted list (by f1 asc):
            # next_f1 = f_{i+1,1} if i < n-1 else ref_1
            # contrib = (next_f1 - p1) * (ref_2 - p2)
            
            # We need to find the index of this point in the sorted list.
            # Since we are calling this for selection, we can pass the sorted list and the index.
            # But here we just have the point. We should pass the sorted list and the index.
            # Let's refactor to accept index.
            
            # For now, this function is a placeholder for the logic, 
            # but in the main loop we will compute contributions for all ND points at once.
            return 0.0

        # Phase 1: Global Sampling
        # Use a mix of random and grid points to ensure coverage
        n_global = 50
        
        # Grid points (corners)
        grid_points = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            lo[0] + bounds_range[0]*i,
                            lo[1] + bounds_range[1]*j,
                            lo[2] + bounds_range[2]*k,
                            lo[3] + bounds_range[3]*l
                        ])
                        grid_points.append(x)
        
        # Random points
        random_points = lo + bounds_range * np.random.random((n_global - len(grid_points), self.dim))
        
        initial_points = np.vstack([grid_points, random_points]) if len(random_points) > 0 else np.array(grid_points)
        
        for x in initial_points:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # Phase 2: Adaptive Local Refinement
        # We will perform multiple rounds of local search on the best non-dominated points.
        
        # Initial step size
        step_scale = 0.2 
        
        # Number of points to refine
        k_refine = 10
        
        # Reference point for HV calculation (from task description)
        ref_point = np.array([3000.0, 0.06])
        
        while eval_count < self.budget and time.time() < deadline:
            nd_indices = get_non_dominated_indices()
            if len(nd_indices) == 0:
                break
            
            nd_indices = np.array(nd_indices)
            
            # Calculate Hypervolume Contribution for each non-dominated point
            # to prioritize parents that contribute most to the current front.
            # We need the ND points sorted by f1.
            nd_F = np.array(all_F)[nd_indices]
            
            # Filter points that are strictly better than ref in both objectives
            valid_mask = (nd_F[:, 0] < ref_point[0]) & (nd_F[:, 1] < ref_point[1])
            valid_nd_F = nd_F[valid_mask]
            valid_nd_indices = nd_indices[valid_mask]
            
            if len(valid_nd_F) == 0:
                # If no points are valid, just pick the best sum
                sums = np.sum(nd_F, axis=1)
                sorted_nd = nd_indices[np.argsort(sums)]
                k = min(k_refine, len(sorted_nd))
                indices_to_refine = sorted_nd[:k]
            else:
                # Sort by f1 ascending
                sort_idx = np.argsort(valid_nd_F[:, 0])
                sorted_F = valid_nd_F[sort_idx]
                sorted_indices = valid_nd_indices[sort_idx]
                
                # Calculate contributions
                # contrib[i] = (next_f1 - curr_f1) * (ref_2 - curr_f2)
                n_valid = len(sorted_F)
                contributions = np.zeros(n_valid)
                
                for i in range(n_valid):
                    curr_f1 = sorted_F[i, 0]
                    curr_f2 = sorted_F[i, 1]
                    
                    if i < n_valid - 1:
                        next_f1 = sorted_F[i+1, 0]
                    else:
                        next_f1 = ref_point[0]
                    
                    # Width in obj 1
                    width = next_f1 - curr_f1
                    # Height in obj 2
                    height = ref_point[1] - curr_f2
                    
                    contributions[i] = width * height
                
                # Select top k_refine based on contribution
                # If contributions are all 0 or very small, fall back to sum
                if np.sum(contributions) < 1e-10:
                     sums = np.sum(valid_nd_F, axis=1)
                     sort_idx = np.argsort(sums)
                     selected_indices = sorted_indices[sort_idx[:k_refine]]
                else:
                     # Sort by contribution descending
                     contrib_sort_idx = np.argsort(contributions)[::-1]
                     selected_indices = sorted_indices[contrib_sort_idx[:k_refine]]
                
                indices_to_refine = selected_indices

            k = len(indices_to_refine)
            if k == 0:
                break
            
            # If we have very little budget left, just do one step for each
            remaining = self.budget - eval_count
            if remaining < k:
                # Just pick the best one and use all budget
                indices_to_refine = [indices_to_refine[0]]
                k = 1
            
            evals_per_point = max(1, remaining // k)
            
            improved = False
            
            for idx in indices_to_refine:
                x_best = all_X[idx].copy()
                f_best = all_F[idx].copy()
                
                # Local Search Loop for this point
                for _ in range(evals_per_point):
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    # Generate candidate with adaptive step
                    step = step_scale * bounds_range * np.random.random(self.dim)
                    direction = np.random.choice([-1, 1], self.dim)
                    candidate = x_best + direction * step
                    candidate = np.clip(candidate, lo, hi)
                    
                    f_cand = evaluate(candidate)
                    if f_cand is None:
                        break
                    
                    # Accept if it dominates the current best
                    if dominates(f_cand, f_best):
                        x_best = candidate
                        f_best = f_cand
                        improved = True
                    else:
                        pass

            # If no improvement in this round, reduce step size
            if not improved:
                step_scale *= 0.5
                if step_scale < 0.001:
                    break
            else:
                pass

        return all_X
