import time
import random
import numpy as np
from collections import Counter

def _calculate_score(points):
    """
    Calculate the maximum multiplicity of any non-zero doubled area.
    points: list of (x, y) tuples
    """
    pts = np.array(points, dtype=np.int32)
    n = len(pts)
    area_counts = Counter()
    
    # Vectorized calculation for all triples is hard to do in one shot for N=30 
    # without a massive temporary array (30*29*28/6 ~ 4000 areas). 
    # But for scoring the initial state, we can just loop or use a helper.
    # For N=30, a simple loop is fine for initialization.
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                dx2 = x3 - x1
                dy2 = y3 - y1
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    area_counts[area] += 1
    
    if not area_counts:
        return 0
    return max(area_counts.values())

def _get_areas_involving(pts, idx):
    """
    Returns a numpy array of absolute doubled areas for all triples involving point idx.
    pts: numpy array of shape (30, 2)
    idx: index of the point
    """
    # Points other than idx
    others = np.delete(pts, idx, axis=0) # Shape (29, 2)
    p_i = pts[idx]
    
    # Vectors from p_i to others
    # V_j = P_j - P_i
    V = others - p_i # Shape (29, 2)
    
    # Cross product for all pairs (j, k)
    # C[j, k] = V[j, 0]*V[k, 1] - V[j, 1]*V[k, 0]
    # We only need j < k
    # Matrix multiplication approach:
    # Let X = V[:, 0], Y = V[:, 1]
    # Cross[j, k] = X[j]*Y[k] - Y[j]*X[k]
    X = V[:, 0]
    Y = V[:, 1]
    
    # Create matrix of cross products
    # M[j, k] = X[j] * Y[k]
    M1 = np.outer(X, Y)
    M2 = np.outer(Y, X)
    Cross = M1 - M2
    
    # Get upper triangle indices
    # np.triu_indices
    j_idx, k_idx = np.triu_indices(29, k=1)
    areas = np.abs(Cross[j_idx, k_idx])
    
    return areas

def _local_search_numpy(initial_points, time_limit=20.0, seed=None):
    """
    Perform hill climbing using NumPy for fast area evaluation.
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    
    # Convert to numpy array
    pts = np.array(initial_points, dtype=np.int32)
    
    # Initial counts
    # We need the full counter to know the current max
    # We can build it by summing areas for each point i, but we must avoid double counting.
    # Easier: Just compute all areas once for initialization.
    area_counts = Counter()
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                dx2 = x3 - x1
                dy2 = y3 - y1
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    area_counts[area] += 1
                    
    current_score = max(area_counts.values()) if area_counts else 0
    best_score = current_score
    best_pts = pts.copy()
    
    occupied_set = set(map(tuple, initial_points))
    
    stuck_count = 0
    max_stuck = 50 # Lower max_stuck because moves are faster, we can restart more often
    
    # Precompute list of all valid coordinates for random fallback
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = tuple(pts[idx])
        
        # Generate candidate new position
        # Strategy: Try nearby positions first, then random
        new_p = None
        # 1. Nearby
        radius = 5
        x, y = old_p
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    p = (nx, ny)
                    if p not in occupied_set:
                        candidates.append(p)
        
        if candidates:
            # Weighted random choice favoring closer points
            weights = []
            for p in candidates:
                dist_sq = (p[0]-x)**2 + (p[1]-y)**2
                weights.append(1.0 / (1.0 + dist_sq))
            new_p = random.choices(candidates, weights=weights, k=1)[0]
        else:
            # Fallback to random empty spot
            empty = [c for c in all_coords if c not in occupied_set]
            if empty:
                new_p = random.choice(empty)
        
        if new_p is None or new_p == old_p:
            continue
            
        # Evaluate move
        # 1. Get old areas involving idx
        old_areas = _get_areas_involving(pts, idx)
        
        # 2. Remove old areas from counts
        for a in old_areas:
            if a > 0:
                area_counts[a] -= 1
                if area_counts[a] == 0:
                    del area_counts[a]
        
        # 3. Apply move
        pts[idx] = new_p
        occupied_set.remove(old_p)
        occupied_set.add(new_p)
        
        # 4. Get new areas involving idx
        new_areas = _get_areas_involving(pts, idx)
        
        # 5. Add new areas to counts
        for a in new_areas:
            if a > 0:
                area_counts[a] += 1
        
        # 6. Check score
        # Optimization: Instead of max(area_counts.values()) which is O(M),
        # we can track it, but M is small. Let's just take max.
        new_score = max(area_counts.values()) if area_counts else 0
        
        if new_score > current_score:
            current_score = new_score
            stuck_count = 0
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            # Revert
            # Remove new areas
            for a in new_areas:
                if a > 0:
                    area_counts[a] -= 1
                    if area_counts[a] == 0:
                        del area_counts[a]
            
            # Restore old point
            pts[idx] = old_p
            occupied_set.remove(new_p)
            occupied_set.add(old_p)
            
            # Restore old areas
            for a in old_areas:
                if a > 0:
                    area_counts[a] += 1
            
            current_score = max(area_counts.values()) if area_counts else 0 # Should be same as before
            
            stuck_count += 1
            if stuck_count > max_stuck:
                # Restart from best
                pts = best_pts.copy()
                occupied_set = set(map(tuple, pts))
                # Rebuild counts
                area_counts = Counter()
                for i in range(n):
                    x1, y1 = pts[i]
                    for j in range(i + 1, n):
                        x2, y2 = pts[j]
                        dx1 = x2 - x1
                        dy1 = y2 - y1
                        for k in range(j + 1, n):
                            x3, y3 = pts[k]
                            dx2 = x3 - x1
                            dy2 = y3 - y1
                            area = abs(dx1 * dy2 - dy1 * dx2)
                            if area > 0:
                                area_counts[area] += 1
                current_score = best_score
                stuck_count = 0
                
                # Perturbation
                if random.random() < 0.7:
                    for _ in range(5):
                        ridx = random.randint(0, 29)
                        old_rp = tuple(pts[ridx])
                        # Move to random nearby
                        x, y = old_rp
                        rad = 8
                        cands = []
                        for dx in range(-rad, rad+1):
                            for dy in range(-rad, rad+1):
                                if dx==0 and dy==0: continue
                                nx, ny = x+dx, y+dy
                                if -30<=nx<=30 and -30<=ny<=30:
                                    p=(nx,ny)
                                    if p not in occupied_set:
                                        cands.append(p)
                        if cands:
                            new_rp = random.choice(cands)
                            pts[ridx] = new_rp
                            occupied_set.remove(old_rp)
                            occupied_set.add(new_rp)
                    # Recalculate counts after perturbation
                    area_counts = Counter()
                    for i in range(n):
                        x1, y1 = pts[i]
                        for j in range(i + 1, n):
                            x2, y2 = pts[j]
                            dx1 = x2 - x1
                            dy1 = y2 - y1
                            for k in range(j + 1, n):
                                x3, y3 = pts[k]
                                dx2 = x3 - x1
                                dy2 = y3 - y1
                                area = abs(dx1 * dy2 - dy1 * dx2)
                                if area > 0:
                                    area_counts[area] += 1
                    current_score = max(area_counts.values()) if area_counts else 0

    return [tuple(p) for p in best_pts], best_score

def _generate_spiral(center_x, center_y, points_needed=30):
    """Generate a spiral of points."""
    points = [(center_x, center_y)]
    dx, dy = 1, 0
    x, y = center_x, center_y
    step_count = 0
    step_limit = 2
    dir_idx = 0
    dirs = [(1,0), (0,1), (-1,0), (0,-1)]
    
    while len(points) < points_needed:
        x += dx
        y += dy
        p = (x, y)
        if -30 <= x <= 30 and -30 <= y <= 30 and p not in points:
            points.append(p)
        
        step_count += 1
        if step_count == step_limit:
            step_count = 0
            dir_idx = (dir_idx + 1) % 4
            dx, dy = dirs[dir_idx]
            if dir_idx % 2 == 0:
                step_limit += 2
    return points[:points_needed]

def construct():
    start_total = time.time()
    deadline = start_total + 55.0 # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Grid 5x6
    grid_5x6 = [(x, y) for x in range(5) for y in range(6)]
    candidates.append(grid_5x6)
    
    # 2. Grid 6x5
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(grid_6x5)
    
    # 3. Spiral
    spiral = _generate_spiral(0, 0, 30)
    candidates.append(spiral)
    
    # 4. Random
    random.seed(42)
    random_points = random.sample([(x, y) for x in range(-30, 31) for y in range(-30, 31)], 30)
    candidates.append(random_points)
    
    # 5. Hex-like
    hex_pts = []
    for i in range(6):
        for j in range(5):
            x = j
            y = i
            if i % 2 == 1:
                x += 1
            hex_pts.append((x, y))
    candidates.append(hex_pts[:30])

    best_overall_points = None
    best_overall_score = -1
    
    # Evaluate candidates to pick the best start for the main search
    for c in candidates:
        if len(c) != 30:
            continue
        s = _calculate_score(c)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = c
            
    # Multi-start local search
    # We have ~55s left.
    # Let's run 3 independent searches with 15s each.
    # Start 1: Best initial candidate
    # Start 2: Perturbed Best initial candidate
    # Start 3: Random
    
    results = []
    
    # Search 1
    t_start = time.time()
    if time.time() < deadline - 15.0:
        p1, s1 = _local_search_numpy(best_overall_points, time_limit=15.0, seed=1)
        results.append((p1, s1))
        
    # Search 2: Perturb the best found
    if time.time() < deadline - 15.0:
        # Perturb best_overall_points
        perturbed = list(best_overall_points)
        random.seed(2)
        for _ in range(10):
            idx = random.randint(0, 29)
            x, y = perturbed[idx]
            # Move to nearby
            rad = 5
            cands = []
            for dx in range(-rad, rad+1):
                for dy in range(-rad, rad+1):
                    if dx==0 and dy==0: continue
                    nx, ny = x+dx, y+dy
                    if -30<=nx<=30 and -30<=ny<=30:
                        p=(nx,ny)
                        if p not in perturbed:
                            cands.append(p)
            if cands:
                perturbed[idx] = random.choice(cands)
        
        p2, s2 = _local_search_numpy(perturbed, time_limit=15.0, seed=2)
        results.append((p2, s2))
        
    # Search 3: Random start
    if time.time() < deadline - 15.0:
        random.seed(3)
        random_points_3 = random.sample([(x, y) for x in range(-30, 31) for y in range(-30, 31)], 30)
        p3, s3 = _local_search_numpy(random_points_3, time_limit=15.0, seed=3)
        results.append((p3, s3))
        
    # Pick the best result
    if results:
        best_res = max(results, key=lambda x: x[1])
        if best_res[1] > best_overall_score:
            return best_res[0]
            
    return best_overall_points

# EVOLVE-BLOCK-END
