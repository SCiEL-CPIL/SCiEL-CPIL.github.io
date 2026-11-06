"""
Optimized search for 30 points maximizing max multiplicity of a doubled area.
Strategy:
1. Fast vectorized area computation.
2. Targeted local search: for a fixed target area A, greedily add points that
   maximize the number of new triangles with area A.
3. Multi-restart with different target areas and initial seeds.
"""
import numpy as np
import random
import time
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # safety margin before 90s limit (evaluator kills at 90s, but task says 60s? Let's use 58 to be safe for 60s limit mentioned in contract)
    # Contract says "Each evaluation must finish within 60 s". Directive says "evaluator KILLS the program after 90s". 
    # I will use 58s to be safe against the 60s limit mentioned in the contract text.

    def compute_areas_vectorized(points):
        """
        Compute all doubled areas for all triples in a vectorized manner.
        Returns a numpy array of positive areas.
        """
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        if n < 3:
            return np.array([], dtype=np.int64)
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        # We need all triples i < j < k
        # To avoid creating huge intermediate arrays of indices, we can iterate over i
        # and vectorize j, k.
        # However, for n=30, full vectorization with meshgrid is fine and fast.
        # Let's stick to the efficient meshgrid approach but ensure we only keep i<j<k.
        
        # Alternative: Iterate i, and for each i, compute areas with all j,k > i.
        # This is O(n^3) but with low constant factor if vectorized over j,k.
        
        all_areas = []
        for i in range(n - 2):
            # Points from i+1 to n-1
            sub_pts = pts[i+1:]
            m = len(sub_pts)
            if m < 2:
                continue
            sx = sub_pts[:, 0]
            sy = sub_pts[:, 1]
            xi = x[i]
            yi = y[i]
            
            # We need pairs (j, k) from sub_pts where index_j < index_k in sub_pts
            # This corresponds to j < k in original indices.
            # Create indices for sub_pts
            idx = np.arange(m)
            # Meshgrid for j, k
            jj, kk = np.meshgrid(idx, idx, indexing='ij')
            # Keep only jj < kk
            mask = jj < kk
            jj_m = jj[mask]
            kk_m = kk[mask]
            
            xj = sx[jj_m]
            yj = sy[jj_m]
            xk = sx[kk_m]
            yk = sy[kk_m]
            
            # Area = |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
            # = |xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj|
            areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
            # Filter positive
            positive = areas[areas > 0]
            if len(positive) > 0:
                all_areas.append(positive)
        
        if not all_areas:
            return np.array([], dtype=np.int64)
        
        return np.concatenate(all_areas)

    def get_max_multiplicity(points):
        """
        Compute the max multiplicity of any positive area.
        Returns (max_count, best_area, area_counts_dict)
        """
        areas = compute_areas_vectorized(points)
        if len(areas) == 0:
            return 0, 0, {}
        
        values, counts = np.unique(areas, return_counts=True)
        max_idx = np.argmax(counts)
        max_count = int(counts[max_idx])
        best_area = int(values[max_idx])
        
        # Build dict for quick lookup if needed, or just return the array
        # For speed in local search, we might just want the max and the area.
        # But for greedy addition, we need to know how many triangles of area A a new point creates.
        # So we need to be able to compute areas involving a specific new point.
        
        return max_count, best_area, dict(zip(values.astype(int), counts.astype(int)))

    def count_triangles_with_area(points, target_area):
        """
        Count number of triples with exactly target_area.
        """
        areas = compute_areas_vectorized(points)
        if len(areas) == 0:
            return 0
        return int(np.sum(areas == target_area))

    def count_new_triangles_with_area(points, new_point, target_area):
        """
        Count how many new triangles with target_area are formed by adding new_point to points.
        A new triangle involves new_point and two points from points.
        """
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        if n < 2:
            return 0
        
        x = pts[:, 0]
        y = pts[:, 1]
        nx, ny = new_point
        
        # We need pairs (i, j) from points such that area((nx,ny), pts[i], pts[j]) == target_area
        # Area = |nx(yi-yj) + xi(yj-ny) + xj(ny-yi)|
        
        # Vectorize over i, j
        # i < j to avoid duplicates? Yes, since triangle is unordered.
        # But we are just counting, so order doesn't matter for the count, 
        # but we must not double count the same pair.
        
        # Create meshgrid
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        mask = ii < jj
        ii_m = ii[mask]
        jj_m = jj[mask]
        
        xi = x[ii_m]
        yi = y[ii_m]
        xj = x[jj_m]
        yj = y[jj_m]
        
        areas = np.abs(nx * (yi - yj) + xi * (yj - ny) + xj * (ny - yi))
        
        return int(np.sum(areas == target_area))

    # Precompute all valid points
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    all_valid_points_arr = np.array(all_valid_points, dtype=np.int64)
    
    best_global_score = 0
    best_global_points = None
    best_global_area = 0
    
    # Candidate target areas to try
    # Small areas are more likely to have high multiplicity.
    # Max area is bounded by box size.
    # Let's try a range of small areas.
    target_areas = list(range(1, 50)) + list(range(50, 100, 5))
    
    # Also, we should try to improve existing configurations.
    # Let's start with a few structured seeds.
    
    seeds = []
    
    # Seed 1: Two rows
    for y0 in range(-5, 6):
        pts = [(x, y0) for x in range(-15, 15)] + [(x, y0+1) for x in range(-15, 15)]
        if len(pts) == 30 and len(set(pts)) == 30:
            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                seeds.append(pts)
                
    # Seed 2: Three rows
    for y0 in range(-5, 6):
        pts = []
        for r in range(3):
            for x in range(-10, 10):
                pts.append((x, y0 + r))
        if len(pts) == 30 and len(set(pts)) == 30:
            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                seeds.append(pts)
                
    # Seed 3: Grid
    for ox in range(-5, 6):
        for oy in range(-5, 6):
            pts = [(ox + x, oy + y) for x in range(6) for y in range(5)]
            if len(pts) == 30 and len(set(pts)) == 30:
                if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                    seeds.append(pts)
                    
    # Evaluate seeds
    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break
        score, area, counts = get_max_multiplicity(seed)
        if score > best_global_score:
            best_global_score = score
            best_global_points = seed[:]
            best_global_area = area
    
    # If no seeds were valid or score is 0, initialize with random
    if best_global_points is None:
        best_global_points = random.sample(all_valid_points, 30)
        best_global_score, best_global_area, _ = get_max_multiplicity(best_global_points)
        
    # Local Search Phase
    # We will perform a greedy local search where we try to replace a point 
    # or add a point (if we allow size to vary? No, size is fixed at 30).
    # So we must replace.
    
    # A more effective local search:
    # 1. Pick a target area A (initially the best found).
    # 2. For each point in the set, try replacing it with a point from the grid
    #    that maximizes the number of triangles with area A.
    # 3. Accept if the total count of area A increases.
    # 4. If stuck, try a different target area.
    
    # To speed up, we can precompute for each candidate point P, 
    # how many triangles of area A it forms with the current set S.
    # But S changes. So we recompute for the current S.
    
    # Optimization:
    # For a fixed S and target A, the "value" of a point P is count_new_triangles_with_area(S, P, A).
    # When we replace P_old with P_new, the change in total count is:
    # Delta = Value(P_new) - Value(P_old) + (interaction between P_new and P_old? No, P_old is removed)
    # Wait, if we replace P_old with P_new:
    # Triangles involving P_old are removed.
    # Triangles involving P_new are added.
    # Triangles not involving P_old or P_new remain.
    # So Delta = Count_new(P_new) - Count_old(P_old).
    # Where Count_old(P_old) is the number of triangles of area A involving P_old in the current set.
    
    # We can compute Count_old(P_old) easily by looking at the current set.
    # And we can compute Count_new(P_new) for all candidate P_new in parallel?
    # Count_new(P_new) depends on the current set S (excluding P_old? No, S includes P_old).
    # When adding P_new, we form triangles with pairs from S \ {P_old}.
    # So we should compute Count_new(P_new) with respect to S \ {P_old}.
    
    # This is slightly complex. Let's simplify:
    # 1. Remove P_old from S to get S'.
    # 2. For each candidate P_new, compute count_new_triangles_with_area(S', P_new, A).
    # 3. Choose P_new that maximizes this count.
    # 4. If max count > count_old(P_old), accept.
    
    # This is O(N * K * M) where N is points in S, K is candidates, M is pairs in S'.
    # N=30, K~3000, M~400. 30 * 3000 * 400 = 36,000,000 ops per iteration.
    # This might be too slow in Python loops.
    # We need to vectorize the "for each candidate P_new" part.
    
    def greedy_improve(points, target_area, max_iters=100):
        """
        Try to improve the set of points by replacing points to maximize
        the number of triangles with target_area.
        """
        current_points = points[:]
        current_set = set(current_points)
        
        # Initial count
        current_count = count_triangles_with_area(current_points, target_area)
        
        # Precompute counts for each point in the current set
        # count_for_point[i] = number of triangles of area target_area involving point i
        # This is the "value" of point i.
        
        def compute_point_values(pts, target):
            values = np.zeros(len(pts), dtype=np.int64)
            n = len(pts)
            pts_arr = np.array(pts, dtype=np.int64)
            x = pts_arr[:, 0]
            y = pts_arr[:, 1]
            
            for i in range(n):
                # Count triangles involving i
                # Pairs (j, k) with j < k, j != i, k != i
                # To vectorize, we can just compute all pairs and mask out i.
                # Or iterate j, k?
                # Let's use the vectorized count_new_triangles_with_area logic but 
                # adapted for an existing point in the set.
                
                # Actually, count_new_triangles_with_area(S, P, A) counts triangles (P, j, k) 
                # where j,k in S.
                # So if we call count_new_triangles_with_area(current_points_without_i, current_points[i], target_area),
                # it gives the value of point i.
                
                # But calling it 30 times is expensive.
                # Let's compute all pairwise areas first?
                # No, let's just use the function. 30 calls is okay if the function is fast.
                
                # To make it faster, we can compute the areas involving i with all pairs (j,k) in one go.
                # But the function count_new_triangles_with_area does exactly that.
                # Let's optimize count_new_triangles_with_area.
                
                pass # Placeholder
            
            # Let's implement a faster version:
            # Compute all areas for all triples? No, that's O(N^3).
            # For each i, we need O(N^2) work. Total O(N^3) = 27000 ops. That's fast!
            
            for i in range(n):
                xi, yi = pts[i]
                # Get indices of all other points
                others = [j for j in range(n) if j != i]
                m = len(others)
                if m < 2:
                    continue
                ox = np.array([pts[j][0] for j in others], dtype=np.int64)
                oy = np.array([pts[j][1] for j in others], dtype=np.int64)
                
                # Pairs from others
                oi, oj = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
                mask = oi < oj
                oi_m = oi[mask]
                oj_m = oj[mask]
                
                x1 = ox[oi_m]
                y1 = oy[oi_m]
                x2 = ox[oj_m]
                y2 = oy[oj_m]
                
                areas = np.abs(xi * (y1 - y2) + x1 * (y2 - yi) + x2 * (yi - y1))
                values[i] = np.sum(areas == target)
            
            return values
        
        # Main loop
        for iter in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            values = compute_point_values(current_points, target_area)
            
            # Find the point with the lowest value
            min_idx = np.argmin(values)
            min_val = values[min_idx]
            
            # Try to replace min_idx with a better point
            # Remove min_idx
            remaining = [p for i, p in enumerate(current_points) if i != min_idx]
            
            # Find the best replacement from all_valid_points
            # We want to maximize count_new_triangles_with_area(remaining, candidate, target_area)
            
            # To speed this up, we can vectorize over candidates.
            # But count_new_triangles_with_area involves a loop over pairs in remaining.
            # Let's precompute the pair areas for remaining?
            # No, the area depends on the candidate point.
            
            # Let's just iterate over a subset of candidates or all candidates if fast enough.
            # 3600 candidates * 400 pairs = 1.44M ops. 
            # In Python, this is ~0.1-0.5s. 
            # With 100 iterations, that's 10-50s. Too slow.
            
            # We need a faster way to find the best replacement.
            # Idea: Randomly sample candidates.
            # Or, use a heuristic: try points near the current set.
            
            # Let's try a limited number of random candidates per iteration.
            num_candidates = 200
            best_candidate = None
            best_candidate_val = -1
            
            # Sample candidates
            candidates = random.sample(all_valid_points, num_candidates)
            
            # Filter out points already in current_points
            candidates = [c for c in candidates if c not in current_set]
            
            for c in candidates:
                val = count_new_triangles_with_area(remaining, c, target_area)
                if val > best_candidate_val:
                    best_candidate_val = val
                    best_candidate = c
            
            if best_candidate is not None and best_candidate_val > min_val:
                # Accept
                current_points[min_idx] = best_candidate
                current_set.discard(current_points[min_idx-1] if min_idx>0 else current_points[-1]) # Update set carefully
                # Actually, just rebuild set or update properly
                current_set = set(current_points)
                current_count = count_triangles_with_area(current_points, target_area)
            else:
                # No improvement found in this iteration
                # Try a few more random replacements?
                # Or break?
                # Let's try 5 more random points to replace the min point
                improved = False
                for _ in range(5):
                    c = random.choice(all_valid_points)
                    if c in current_set:
                        continue
                    val = count_new_triangles_with_area(remaining, c, target_area)
                    if val > min_val:
                        current_points[min_idx] = c
                        current_set = set(current_points)
                        current_count = count_triangles_with_area(current_points, target_area)
                        improved = True
                        break
                if not improved:
                    break
        
        return current_points, current_count

    # Run greedy improve for a few target areas
    # Start with the best area found so far
    target_list = [best_global_area]
    # Add some small areas
    for a in range(1, 20):
        if a not in target_list:
            target_list.append(a)
            
    random.shuffle(target_list)
    
    for target in target_list:
        if time.time() - start_time > DEADLINE:
            break
        
        # Start from best_global_points
        improved_points, improved_count = greedy_improve(best_global_points, target, max_iters=50)
        
        # Check if the overall max multiplicity improved
        score, area, counts = get_max_multiplicity(improved_points)
        if score > best_global_score:
            best_global_score = score
            best_global_points = improved_points
            best_global_area = area
            
    # Final polish: try to improve with the current best area
    if time.time() - start_time < DEADLINE - 5:
        improved_points, _ = greedy_improve(best_global_points, best_global_area, max_iters=50)
        score, area, _ = get_max_multiplicity(improved_points)
        if score > best_global_score:
            best_global_score = score
            best_global_points = improved_points
            best_global_area = area
            
    return best_global_points
