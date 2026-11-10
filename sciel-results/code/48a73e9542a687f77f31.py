import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0
    
    # Pre-compute all valid points in [-30, 30]^2
    all_points = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int64)
    # Correction: range syntax error in thought, fixing in code below
    all_points = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int64)
    num_points = len(all_points)
    
    # Map point tuple to index for O(1) lookup
    point_to_idx = {tuple(p): i for i, p in enumerate(all_points)}

    def get_area_histogram(pts):
        """
        Computes the full histogram of doubled areas for a set of 30 points.
        Returns a defaultdict(int) mapping area -> count.
        """
        counts = defaultdict(int)
        n = len(pts)
        arr = np.array(pts, dtype=np.int64)
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                # Vectorized area calculation for triangles (i, j, k) for all k > j
                # Area = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
                # Doubled Area = | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
                dx1 = xj - xi
                dy1 = yj - yi
                dx2 = rest[:, 0] - xi
                dy2 = rest[:, 1] - yi
                doubled_areas = np.abs(dx1 * dy2 - dy1 * dx2)
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += int(c)
        return counts

    def get_max_score(hist):
        if not hist:
            return 0, 0
        max_count = max(hist.values())
        # In case of tie, pick the smaller area? Or arbitrary. Let's pick smallest for stability.
        max_area = min([k for k, v in hist.items() if v == max_count])
        return max_count, max_area

    def incremental_swap_delta(pts, old_idx, new_pt, hist):
        """
        Calculates the change in the area histogram when replacing pts[old_idx] with new_pt.
        Returns a dictionary of deltas: {area: delta_count}
        """
        deltas = defaultdict(int)
        n = len(pts)
        others = [pts[i] for i in range(n) if i != old_idx]
        
        old_pt = pts[old_idx]
        
        # Remove contributions of old_pt
        # For every pair (i, j) in others, area(old_pt, i, j) decreases by 1
        # For every i in others, area(old_pt, i, new_pt) is NOT involved in removal, 
        # but area(old_pt, i, j) IS.
        # Actually, triangles involving old_pt are:
        # 1. (old_pt, i, j) where i, j are in others.
        
        # Add contributions of new_pt
        # Triangles involving new_pt are:
        # 1. (new_pt, i, j) where i, j are in others.
        
        # Note: Triangles NOT involving old_pt or new_pt (i.e., (i, j, k) all in others) are unchanged.
        
        # To be efficient, we compute the histogram of areas for triangles involving old_pt with others,
        # and subtract it. Then compute for new_pt and add it.
        
        if len(others) < 2:
            return deltas

        others_arr = np.array(others, dtype=np.int64)
        
        def compute_involving_hist(pt, others_arr):
            local_hist = defaultdict(int)
            ox, oy = pt
            dx = others_arr[:, 0] - ox
            dy = others_arr[:, 1] - oy
            # Cross product matrix C where C[i,j] = dx[i]*dy[j] - dy[i]*dx[j]
            C = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
            abs_C = np.abs(C)
            # We only want i < j
            i_idx, j_idx = np.triu_indices(len(others_arr), k=1)
            valid_areas = abs_C[i_idx, j_idx]
            valid_areas = valid_areas[valid_areas > 0]
            if len(valid_areas) > 0:
                vals, cnts = np.unique(valid_areas, return_counts=True)
                for v, c in zip(vals, cnts):
                    local_hist[int(v)] += int(c)
            return local_hist

        old_hist = compute_involving_hist(old_pt, others_arr)
        new_hist = compute_involving_hist(new_pt, others_arr)
        
        # Apply deltas
        for area, count in old_hist.items():
            deltas[area] -= count
        for area, count in new_hist.items():
            deltas[area] += count
            
        return deltas

    # Multi-target search
    # We will run local search for a set of candidate target areas.
    # However, the objective is to maximize the *global* max multiplicity.
    # So we can just run local search optimizing for the current best area, 
    # but periodically re-evaluate the global max to switch targets if a different area overtakes it.
    
    # To improve on the seed, we will use a "best-of-N" parallel search strategy.
    # We will maintain several "chains" of local search, each starting from a different seed
    # and potentially locking onto different areas.
    
    seeds = []
    
    # 1. Structured Seeds
    # Small grids
    for w in range(3, 10):
        for h in range(3, 10):
            if w * h < 30: continue
            # Try to fit a w x h grid in [-30, 30]
            # Max span 60.
            max_dx = 60 // (w - 1) if w > 1 else 60
            max_dy = 60 // (h - 1) if h > 1 else 60
            # Sample a few spacings
            for dx in [1, 2, 3, 4, 5]:
                if dx > max_dx: continue
                for dy in [1, 2, 3, 4, 5]:
                    if dy > max_dy: continue
                    # Offset
                    for ox in range(-10, 11, 5):
                        for oy in range(-10, 11, 5):
                            pts = []
                            valid = True
                            for i in range(w):
                                for j in range(h):
                                    x = ox + i * dx
                                    y = oy + j * dy
                                    if -30 <= x <= 30 and -30 <= y <= 30:
                                        pts.append((x, y))
                                    else:
                                        valid = False
                                        break
                                if not valid: break
                            if valid and len(pts) == 30:
                                seeds.append(pts)
    
    # 2. Random Seeds
    random.seed(42)
    for _ in range(20):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        seeds.append(list(pts))
        
    # 3. "Line Cluster" Seeds: Points close to a line
    for _ in range(10):
        pts = []
        base_x = random.randint(-20, 20)
        base_y = random.randint(-20, 20)
        slope_num = random.randint(-2, 2)
        slope_den = random.randint(1, 3)
        for i in range(30):
            x = base_x + i * 2
            if x > 30: x = 30
            if x < -30: x = -30
            y = base_y + int(i * slope_num / slope_den)
            if y > 30: y = 30
            if y < -30: y = -30
            # Add jitter
            if random.random() < 0.5:
                x += random.randint(-1, 1)
                y += random.randint(-1, 1)
            x = max(-30, min(30, x))
            y = max(-30, min(30, y))
            if (x, y) not in pts:
                pts.append((x, y))
        # Fill remaining randomly
        while len(pts) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        seeds.append(pts)

    # Deduplicate
    unique_seeds = []
    seen = set()
    for s in seeds:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)
            
    best_overall_score = 0
    best_overall_pts = None
    best_overall_hist = None
    
    # Process seeds with local search
    # We will allocate time per seed based on number of seeds
    num_seeds = len(unique_seeds)
    time_per_seed = 40.0 / num_seeds if num_seeds > 0 else 40.0
    time_per_seed = max(1.0, min(5.0, time_per_seed)) # Cap between 1s and 5s per seed
    
    for seed in unique_seeds:
        if time.time() > total_deadline - 5:
            break
            
        current_pts = list(seed)
        current_set = set(current_pts)
        
        # Initial Score
        cur_hist = get_area_histogram(current_pts)
        cur_score, cur_area = get_max_score(cur_hist)
        
        if cur_score > best_overall_score:
            best_overall_score = cur_score
            best_overall_pts = current_pts
            best_overall_hist = cur_hist
            
        # Local Search
        # We will optimize for the current best area in the histogram.
        # If the max area changes, we switch target.
        target_area = cur_area
        
        improved = True
        max_iterations = 100 # Fixed budget per seed, adjusted by time check
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < total_deadline - 5:
            improved = False
            iter_count += 1
            
            # Pick a random index to swap
            idx = random.randint(0, 29)
            old_p = current_pts[idx]
            
            # Generate candidates
            # We want to find a point that increases the count for target_area
            # OR increases the global max.
            # Since global max is expensive to track incrementally for arbitrary changes,
            # we focus on increasing the count for the *current* target_area.
            # We will sample candidates and pick the one with best delta for target_area.
            
            # To speed up, we can pre-filter candidates? 
            # Let's just sample 50 random points from the grid.
            # We can also bias sampling towards points that are "geometrically similar" to the set?
            # For now, random sampling from the whole grid.
            
            candidates = random.sample(range(num_points), 50)
            
            best_candidate = None
            best_gain = 0
            best_delta_hist = None
            
            # Calculate current count of target_area triangles involving old_p
            # We can compute this from the delta function logic, but we need the "old" part.
            # Actually, incremental_swap_delta returns the net change.
            # Gain = delta[target_area]
            
            # Optimization: We can compute the delta for all candidates in a vectorized way?
            # No, the histogram update is discrete.
            # But we can compute the "involving count" for target_area for old_p once.
            
            others = [current_pts[i] for i in range(30) if i != idx]
            others_arr = np.array(others, dtype=np.int64)
            
            def count_involving_target(pt, others_arr, t_area):
                if len(others_arr) < 2: return 0
                ox, oy = pt
                dx = others_arr[:, 0] - ox
                dy = others_arr[:, 1] - oy
                C = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
                abs_C = np.abs(C)
                i_idx, j_idx = np.triu_indices(len(others_arr), k=1)
                valid_areas = abs_C[i_idx, j_idx]
                return np.sum(valid_areas == t_area)
                
            old_target_count = count_involving_target(old_p, others_arr, target_area)
            
            for c_idx in candidates:
                new_p = tuple(all_points[c_idx])
                if new_p in current_set:
                    continue
                new_target_count = count_involving_target(new_p, others_arr, target_area)
                gain = new_target_count - old_target_count
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = new_p
                    # We don't need the full delta hist for selection, just the gain for target
                    # But if we accept, we need to update the hist.
            
            if best_candidate is not None:
                # Perform swap
                current_pts[idx] = best_candidate
                current_set.remove(old_p)
                current_set.add(best_candidate)
                
                # Update Histogram
                # We need the full delta to update the histogram correctly
                # This is the expensive part. 
                # Can we avoid full delta?
                # The histogram changes only for areas that appear in triangles involving old_p or new_p.
                # We can compute the delta dict.
                delta_hist = incremental_swap_delta(current_pts, idx, best_candidate, cur_hist)
                # Wait, incremental_swap_delta uses current_pts which has already been swapped?
                # No, I passed current_pts (which is modified) and idx and new_pt.
                # My function `incremental_swap_delta` takes `pts` (current list), `old_idx` (index to replace), `new_pt`.
                # It calculates deltas based on `pts[old_idx]` (which is now new_pt) and `new_pt`?
                # No, that's wrong. `pts[old_idx]` is the NEW point now.
                # I need to pass the OLD point or calculate before swap.
                
                # Let's refactor: Calculate delta BEFORE swap.
                # Undo swap temporarily or just pass old_p explicitly.
                
                # Let's rewrite the update logic to be safer.
                # We will re-calculate the histogram if the gain is significant? No, too slow.
                # Let's fix `incremental_swap_delta` to take `old_pt` and `new_pt` and `others`.
                
                # Actually, I can just compute the delta dict in the loop above?
                # No, that's too much overhead.
                
                # Let's use a slightly different approach for histogram update:
                # 1. Compute areas involving old_p with others.
                # 2. Compute areas involving new_p with others.
                # 3. Update hist.
                
                # This is what `incremental_swap_delta` does, but I need to pass the correct arguments.
                # Let's create a helper that takes old_pt, new_pt, others_arr.
                
                def compute_delta_hist(old_pt, new_pt, others_arr):
                    deltas = defaultdict(int)
                    if len(others_arr) < 2: return deltas
                    ox, oy = old_pt
                    dx = others_arr[:, 0] - ox
                    dy = others_arr[:, 1] - oy
                    C_old = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
                    abs_C_old = np.abs(C_old)
                    i_idx, j_idx = np.triu_indices(len(others_arr), k=1)
                    areas_old = abs_C_old[i_idx, j_idx]
                    areas_old = areas_old[areas_old > 0]
                    if len(areas_old) > 0:
                        vals, cnts = np.unique(areas_old, return_counts=True)
                        for v, c in zip(vals, cnts):
                            deltas[int(v)] -= int(c)
                            
                    nx, ny = new_pt
                    dx = others_arr[:, 0] - nx
                    dy = others_arr[:, 1] - ny
                    C_new = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
                    abs_C_new = np.abs(C_new)
                    areas_new = abs_C_new[i_idx, j_idx]
                    areas_new = areas_new[areas_new > 0]
                    if len(areas_new) > 0:
                        vals, cnts = np.unique(areas_new, return_counts=True)
                        for v, c in zip(vals, cnts):
                            deltas[int(v)] += int(c)
                            
                    return deltas

                delta_hist = compute_delta_hist(old_p, best_candidate, others_arr)
                
                for area, d in delta_hist.items():
                    cur_hist[area] += d
                    if cur_hist[area] == 0:
                        del cur_hist[area]
                        
                improved = True
                
                # Check if target area should be updated
                # We only re-evaluate max score every 10 iterations to save time
                if iter_count % 10 == 0 or iter_count == max_iterations:
                    new_score, new_area = get_max_score(cur_hist)
                    if new_score > best_overall_score:
                        best_overall_score = new_score
                        best_overall_pts = current_pts
                        best_overall_hist = cur_hist
                    # Update target if the max area has changed
                    if new_area != target_area:
                        target_area = new_area

    # Final check
    if best_overall_pts is None:
        # Fallback to a simple grid
        pts = []
        for i in range(30):
            x = i % 6
            y = i // 6
            pts.append((x*5-15, y*5-10))
        best_overall_pts = pts
        
    return best_overall_pts
