"""Multi-objective optimizer on WFG4 using Hypervolume-Indicating MOEA with Island Model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)
            
        # Island Model Parameters
        n_islands = 4
        pop_size = 10  # Per island
        total_pop = n_islands * pop_size
        
        # Initialize islands with random populations
        islands = []
        for i in range(n_islands):
            # Random initialization within bounds
            X = np.random.uniform(lo, hi, (pop_size, dim))
            F = np.array([problem(x) for x in X])
            islands.append({'X': X, 'F': F})
            
        used = total_pop
        
        # Helper: Fast 3D Hypervolume Contribution
        # Reference point for contribution calculation should be a point worse than all in the set.
        # We use the reference point from the task [3, 5, 7] as a safe upper bound for "worse".
        # Actually, for relative contribution within a population, we can use the max of the population.
        # But to be consistent with the scoring metric, let's stick to the global ref if possible,
        # or just use a local ref. The ranking is what matters.
        # Let's use a local ref: max of F + small epsilon.
        
        def calc_hv_contribution(F_set, ref_point):
            """
            Calculate hypervolume contribution for each point in F_set (n x 3).
            ref_point: (3,)
            Returns: array of contributions (n,)
            """
            n = len(F_set)
            if n == 0:
                return np.array([])
            if n == 1:
                # Vol = prod(ref - f)
                return np.array([np.prod(ref_point - F_set[0])])
            
            contributions = np.zeros(n)
            
            # For 3D, we can compute exact HV of subsets or use the "additive" property.
            # HV(S) = HV(S \ {i}) + Contribution(i)
            # This is O(n^2) or O(n^3) if done naively.
            # For small n (10), O(n^3) is 1000 ops, which is fine.
            # But we can do better.
            
            # Sort by first objective
            idx = np.argsort(F_set[:, 0])
            F_sorted = F_set[idx]
            
            # We need to calculate the volume of the union of boxes defined by each point and the reference.
            # Box i is [F_i[0], ref[0]] x [F_i[1], ref[1]] x [F_i[2], ref[2]]
            # The hypervolume is the volume of the union of these boxes.
            
            # Contribution of point i is the volume of Box_i minus the volume of the union of boxes j where j dominates i?
            # No, standard decomposition:
            # Sort by f0.
            # The contribution of the i-th point (in sorted order) is the volume of the slice of the union
            # that is "owned" by this point.
            
            # Let's use a simpler recursive or direct calculation for n=10.
            # Brute force: Calculate HV of full set. Then for each i, calculate HV of set without i.
            # Contribution_i = HV(All) - HV(All \ {i})
            # This is O(n * HV_cost). HV_cost for 10 points in 3D is small.
            
            def hv_3d(points, ref):
                """Exact HV for 3D points vs ref."""
                if len(points) == 0:
                    return 0.0
                if len(points) == 1:
                    return np.prod(ref - points[0])
                
                # Sort by first objective
                idx = np.argsort(points[:, 0])
                pts = points[idx]
                
                # The HV is the sum of volumes of "slabs"
                # For each point i, it contributes the volume of the box [pts[i][0], pts[i+1][0]] (or ref[0]) 
                # times the 2D HV of the points that are "active" in that slab.
                
                total_vol = 0.0
                # Process points in order of increasing f0
                # Active points are those with f0 <= current_x.
                # We need the 2D HV of the projections of active points onto (f1, f2) plane.
                
                # To make it efficient, we can iterate through the unique f0 values.
                # Since n is small, we can just do:
                # For each interval [x_i, x_{i+1}], the volume is (x_{i+1} - x_i) * HV_2D({p_j | p_j[0] <= x_i}, ref[1:])
                
                # Let's implement HV_2D
                def hv_2d(pts_2d, ref_2d):
                    if len(pts_2d) == 0:
                        return 0.0
                    if len(pts_2d) == 1:
                        return np.prod(ref_2d - pts_2d[0])
                    
                    # Sort by first dim
                    idx2 = np.argsort(pts_2d[:, 0])
                    p = pts_2d[idx2]
                    
                    vol = 0.0
                    # Iterate through points.
                    # The area is sum of (width * height)
                    # For 2D, the HV is the area under the "staircase" formed by the points.
                    # Sort by f1.
                    # Area = sum_{i} (p[i][0] - p[i-1][0]) * (ref[1] - min_{j<=i} p[j][1]) ? No.
                    
                    # Standard 2D HV:
                    # Sort by f1 ascending.
                    # Initialize min_f2 = ref[1]
                    # Area = 0
                    # For each point in sorted order:
                    #   Area += (min_f2 - p[i][1]) * (ref[0] - p[i][0]) ? No.
                    
                    # Let's use the standard algorithm:
                    # Sort by f1.
                    # Keep track of the minimum f2 seen so far? No, that's for single objective.
                    # For multi-objective, we need the union area.
                    
                    # Simple O(n^2) for 2D:
                    # Area = sum over i of (ref[0] - p[i][0]) * (ref[1] - max_{j: p[j][0] <= p[i][0]} p[j][1])?
                    # No.
                    
                    # Let's just use the "slice" method.
                    # Sort by f1.
                    # The area is sum_{i=0}^{n-1} (p[i+1][0] - p[i][0]) * (ref[1] - min_{j<=i} p[j][1]) + (ref[0] - p[n-1][0]) * (ref[1] - min_{j} p[j][1])
                    # Wait, the "min f2" logic is for the boundary of the union.
                    # The union boundary in 2D is a monotonic path.
                    # Sort by f1.
                    # min_f2_so_far = ref[1]
                    # area = 0
                    # for i in range(n):
                    #   if p[i][1] < min_f2_so_far:
                    #      # This point extends the boundary
                    #      # The area added is (ref[0] - p[i][0]) * (min_f2_so_far - p[i][1])
                    #      area += (ref[0] - p[i][0]) * (min_f2_so_far - p[i][1])
                    #      min_f2_so_far = p[i][1]
                    #      # Plus the area to the left of this point?
                    #      # The area to the left is (p[i][0] - prev_x) * min_f2_so_far?
                    
                    # Let's do it properly.
                    # The hypervolume is the area of the union of rectangles [p[i][0], ref[0]] x [p[i][1], ref[1]].
                    # Sort by p[i][0].
                    # Let x_0 = p[0][0], x_1 = p[1][0], ..., x_n = ref[0].
                    # In the interval [x_i, x_{i+1}], the "active" points are those with p[j][0] <= x_i.
                    # The height of the union in this interval is ref[1] - min_{j active} p[j][1].
                    # So Area = sum_{i=0}^{n-1} (x_{i+1} - x_i) * (ref[1] - min_{j: p[j][0] <= x_i} p[j][1])
                    
                    p_sorted = p
                    min_f2 = ref_2d[1]
                    area = 0.0
                    prev_x = 0.0 # Not needed if we just sum slices
                    for i in range(len(p_sorted)):
                        x_start = p_sorted[i][0]
                        x_end = p_sorted[i+1][0] if i+1 < len(p_sorted) else ref_2d[0]
                        # Update min_f2 with current point
                        if p_sorted[i][1] < min_f2:
                            min_f2 = p_sorted[i][1]
                        area += (x_end - x_start) * (ref_2d[1] - min_f2)
                    return area

                # Back to HV_3D
                # Sort by f0
                p = pts
                min_f1 = ref[1]
                min_f2 = ref[2]
                # This simple min tracking is NOT correct for 3D union.
                # We need HV_2D of the active set.
                
                total_vol = 0.0
                # Active set is empty initially.
                # We iterate through unique f0 values.
                # Since n is small, we can just loop i from 0 to n-1.
                # Interval [p[i][0], p[i+1][0]]
                # Active points: all j such that p[j][0] <= p[i][0].
                # This is just p[0..i].
                # So we need HV_2D of p[0..i] projected to (f1, f2) vs ref[1:].
                
                for i in range(len(p)):
                    x_start = p[i][0]
                    x_end = p[i+1][0] if i+1 < len(p) else ref[0]
                    width = x_end - x_start
                    if width <= 0:
                        continue
                    
                    # Active points are p[0..i]
                    active_2d = p[0:i+1, 1:3]
                    vol_2d = hv_2d(active_2d, ref[1:])
                    total_vol += width * vol_2d
                    
                return total_vol
            
            # Calculate Full HV
            hv_full = hv_3d(F_set, ref_point)
            
            # Calculate contributions
            for i in range(n):
                # Remove i
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                F_reduced = F_set[mask]
                hv_reduced = hv_3d(F_reduced, ref_point)
                contributions[i] = hv_full - hv_reduced
                
            # Map back to original indices
            orig_contribs = np.zeros(n)
            orig_contribs[idx] = contributions
            return orig_contribs

        # Main Loop
        generations = 0
        while used < budget:
            if time.time() > deadline:
                break
                
            # 1. Selection and Variation
            for i in range(n_islands):
                if used >= budget or time.time() > deadline:
                    break
                    
                X = islands[i]['X']
                F = islands[i]['F']
                
                # Calculate HV contributions for this island's population
                # Use a local reference point: max of F + epsilon
                # Or use the global ref [3,5,7] if all points are better.
                # To be safe, use max(F) + 0.1
                local_ref = np.max(F, axis=0) + 0.1
                # Ensure local_ref is not worse than global ref?
                # Actually, for ranking within the island, any consistent ref works.
                # Using local ref makes contributions sensitive to the current spread.
                
                contribs = calc_hv_contribution(F, local_ref)
                
                # Select parents based on contribution
                # Use roulette wheel or top-k.
                # Let's use top-k selection with replacement, weighted by contribution.
                # Or simple: pick the best 2 parents for each child.
                
                # For DE/rand/1/bin:
                # Pick 3 distinct random parents.
                # x_trial = x_r1 + F * (x_r2 - x_r3)
                
                new_X = []
                new_F = []
                
                # Generate pop_size new individuals
                for _ in range(pop_size):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Pick 3 distinct indices
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    
                    # Mutation
                    v = X[r1] + 0.5 * (X[r2] - X[r3]) # F=0.5
                    
                    # Error Repair: Clip to bounds
                    # This is the "error repair" strategy.
                    v = np.clip(v, lo, hi)
                    
                    # Evaluate
                    f_v = problem(v)
                    used += 1
                    new_X.append(v)
                    new_F.append(f_v)
                    
                if not new_X:
                    continue
                    
                new_X = np.array(new_X)
                new_F = np.array(new_F)
                
                # 2. Selection: Combine old and new, select best pop_size
                X_combined = np.vstack([X, new_X])
                F_combined = np.vstack([F, new_F])
                
                # Remove dominated points?
                # Or just select best by HV contribution again?
                # Let's select best pop_size by HV contribution from the combined set.
                
                # Calculate contributions for combined set
                local_ref_comb = np.max(F_combined, axis=0) + 0.1
                contribs_comb = calc_hv_contribution(F_combined, local_ref_comb)
                
                # Select top pop_size
                # If there are ties or zeros, we might pick dominated points?
                # HV contribution is 0 for dominated points (in the union sense, they add 0 volume).
                # So this naturally filters dominated points.
                
                # Get indices of top pop_size contributions
                # If more than pop_size have non-zero contribution, pick the best ones.
                # If fewer, we need to fill the rest.
                
                # Sort indices by contribution descending
                idx_sorted = np.argsort(-contribs_comb)
                
                # Take top pop_size
                selected_idx = idx_sorted[:pop_size]
                
                islands[i]['X'] = X_combined[selected_idx]
                islands[i]['F'] = F_combined[selected_idx]
                
            # 3. Migration
            # After each generation, migrate best individuals between islands.
            # Simple: Swap one best individual between adjacent islands (ring topology).
            for i in range(n_islands):
                j = (i + 1) % n_islands
                # Find best individual in island i (highest HV contribution? or just random?)
                # Let's swap the "best" individual.
                # Define best as the one with highest HV contribution in its island.
                # Or just random swap to maintain diversity.
                # Let's do random swap.
                if np.random.rand() < 0.5: # 50% chance to migrate
                    idx_i = np.random.randint(pop_size)
                    idx_j = np.random.randint(pop_size)
                    
                    # Swap
                    tmp_x = islands[i]['X'][idx_i].copy()
                    tmp_f = islands[i]['F'][idx_i].copy()
                    
                    islands[i]['X'][idx_i] = islands[j]['X'][idx_j]
                    islands[i]['F'][idx_i] = islands[j]['F'][idx_j]
                    
                    islands[j]['X'][idx_j] = tmp_x
                    islands[j]['F'][idx_j] = tmp_f
                    
            generations += 1
            
        # Final Result: Return any point (ignored), but ensure we don't crash.
        # The evaluator records all returned objective vectors from problem calls?
        # "Every returned objective vector is recorded" -> This implies the evaluator wraps problem()
        # and records every f returned. So we just need to make sure we call problem() correctly.
        # The return value of __call__ is ignored.
        
        return None

# EVOLVE-BLOCK-END
