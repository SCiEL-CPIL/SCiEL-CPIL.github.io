import time
import random
import numpy as np

# Precompute all triple indices for 30 points
_N = 30
_TRIPLE_IDX = None

def _get_triple_indices():
    """Precompute indices for all C(30,3) triples."""
    global _TRIPLE_IDX
    if _TRIPLE_IDX is None:
        triples = []
        for i in range(_N):
            for j in range(i + 1, _N):
                for k in range(j + 1, _N):
                    triples.append((i, j, k))
        _TRIPLE_IDX = np.array(triples, dtype=np.int32)
    return _TRIPLE_IDX

def _score_fast(points):
    """
    Fast scoring using precomputed triple indices and NumPy.
    points: list of tuples (x, y)
    Returns: max multiplicity of a non-zero doubled area.
    """
    P = np.array(points, dtype=np.int32)
    triples = _get_triple_indices()
    
    # Extract point coordinates for each triple
    i_idx = triples[:, 0]
    j_idx = triples[:, 1]
    k_idx = triples[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    # Compute doubled area: |det([Pj-Pi, Pk-Pi])|
    # det = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    dx1 = Pj[:, 0] - Pi[:, 0]
    dy1 = Pj[:, 1] - Pi[:, 1]
    dx2 = Pk[:, 0] - Pi[:, 0]
    dy2 = Pk[:, 1] - Pi[:, 1]
    dets = np.abs(dx1 * dy2 - dy1 * dx2)
    
    # Filter non-zero
    mask = dets > 0
    if not np.any(mask):
        return 0
    dets = dets[mask]
    
    # Use bincount for fast histogram
    max_d = int(np.max(dets))
    hist_size = max_d + 1
    counts = np.bincount(dets, minlength=hist_size)
    
    return int(np.max(counts))

def _score_area(points, target_area):
    """
    Count how many triples have exactly the given doubled area.
    """
    P = np.array(points, dtype=np.int32)
    triples = _get_triple_indices()
    
    i_idx = triples[:, 0]
    j_idx = triples[:, 1]
    k_idx = triples[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    dx1 = Pj[:, 0] - Pi[:, 0]
    dy1 = Pj[:, 1] - Pi[:, 1]
    dx2 = Pk[:, 0] - Pi[:, 0]
    dy2 = Pk[:, 1] - Pi[:, 1]
    dets = np.abs(dx1 * dy2 - dy1 * dx2)
    
    return int(np.sum(dets == target_area))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    best_score = 0
    best_points = None
    
    # 1. Systematic Grid Search
    # Factors of 30
    factors = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]
    
    # Spacings
    # Max width 61 points in [-30, 30].
    # If spacing is s, width is (W-1)*s.
    # (W-1)*s <= 60.
    
    for W, H in factors:
        # Max spacing for W
        max_sx = 60 // (W - 1) if W > 1 else 60
        max_sy = 60 // (H - 1) if H > 1 else 60
        
        for sx in range(1, max_sx + 1):
            for sy in range(1, max_sy + 1):
                grid_width = (W - 1) * sx
                grid_height = (H - 1) * sy
                
                if grid_width <= 60 and grid_height <= 60:
                    # Generate points
                    points = [(x * sx, y * sy) for x in range(W) for y in range(H)]
                    
                    score = _score_fast(points)
                    if score > best_score:
                        best_score = score
                        best_points = points[:]
                    
    if best_points is None:
        # Fallback
        best_points = [(i, 0) for i in range(30)]
        best_score = _score_fast(best_points)
        
    # Time check
    if time.time() > deadline - 10:
        return best_points
    
    # 2. Local Search with precomputed collision set
    current_points = best_points[:]
    current_score = best_score
    n = 30
    
    # Precompute neighbor offsets for different radii
    neighbors_r2 = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if not (dx == 0 and dy == 0)]
    neighbors_r3 = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if not (dx == 0 and dy == 0)]
    
    # Hill climbing with multiple radii
    for radius_idx, neighbors in enumerate([neighbors_r2, neighbors_r3, neighbors_r2]):
        if time.time() > deadline - 5:
            break
            
        max_rounds = 20 if radius_idx == 0 else 10
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Use a set for O(1) collision checks
            point_set = set(current_points)
            
            for idx in range(n):
                if time.time() > deadline - 3:
                    break
                    
                ox, oy = current_points[idx]
                
                for dx, dy in neighbors:
                    nx = ox + dx
                    ny = oy + dy
                    
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    
                    # Check collision
                    if (nx, ny) in point_set:
                        continue
                    
                    # Apply move
                    old_point = current_points[idx]
                    current_points[idx] = (nx, ny)
                    point_set.remove(old_point)
                    point_set.add((nx, ny))
                    
                    new_score = _score_fast(current_points)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points[:]
                    else:
                        # Revert
                        current_points[idx] = old_point
                        point_set.remove((nx, ny))
                        point_set.add(old_point)
                        
            if not improved:
                break
    
    # 3. Targeted area-matching search: find the dominant area, then try to
    #    increase its count by moving individual points.
    if time.time() < deadline - 10:
        current_points = best_points[:]
        current_score = best_score
        point_set = set(current_points)
        
        # Find the current dominant area
        P = np.array(current_points, dtype=np.int32)
        triples = _get_triple_indices()
        i_idx = triples[:, 0]
        j_idx = triples[:, 1]
        k_idx = triples[:, 2]
        Pi = P[i_idx]
        Pj = P[j_idx]
        Pk = P[k_idx]
        dx1 = Pj[:, 0] - Pi[:, 0]
        dy1 = Pj[:, 1] - Pi[:, 1]
        dx2 = Pk[:, 0] - Pi[:, 0]
        dy2 = Pk[:, 1] - Pi[:, 1]
        dets = np.abs(dx1 * dy2 - dy1 * dx2)
        mask = dets > 0
        dets_nz = dets[mask]
        
        # Get top few area values to target
        unique_areas, area_counts = np.unique(dets_nz, return_counts=True)
        # Sort by count descending, take top 5
        sorted_idx = np.argsort(area_counts)[::-1]
        top_areas = [int(unique_areas[i]) for i in sorted_idx[:5]]
        
        # For each target area, try to increase its count via single-point moves
        for target_area in top_areas:
            if time.time() > deadline - 5:
                break
            
            target_count = _score_area(current_points, target_area)
            
            for _ in range(10):
                if time.time() > deadline - 3:
                    break
                improved = False
                for idx in range(n):
                    if time.time() > deadline - 2:
                        break
                    ox, oy = current_points[idx]
                    # Try small moves
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            nx = ox + dx
                            ny = oy + dy
                            if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                                continue
                            if (nx, ny) in point_set:
                                continue
                            
                            old_point = current_points[idx]
                            current_points[idx] = (nx, ny)
                            point_set.remove(old_point)
                            point_set.add((nx, ny))
                            
                            new_target_count = _score_area(current_points, target_area)
                            
                            if new_target_count > target_count:
                                target_count = new_target_count
                                improved = True
                                # Check if overall score improved
                                overall = _score_fast(current_points)
                                if overall > best_score:
                                    best_score = overall
                                    best_points = current_points[:]
                            else:
                                current_points[idx] = old_point
                                point_set.remove((nx, ny))
                                point_set.add(old_point)
                
                if not improved:
                    break
    
    # 4. Final polish: exhaustive ±1 moves
    if time.time() < deadline - 3:
        current_points = best_points[:]
        current_score = best_score
        point_set = set(current_points)
        
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            improved = False
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                ox, oy = current_points[idx]
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = ox + dx
                        ny = oy + dy
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        if (nx, ny) in point_set:
                            continue
                        
                        old_point = current_points[idx]
                        current_points[idx] = (nx, ny)
                        point_set.remove(old_point)
                        point_set.add((nx, ny))
                        
                        new_score = _score_fast(current_points)
                        
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points[:]
                        else:
                            current_points[idx] = old_point
                            point_set.remove((nx, ny))
                            point_set.add(old_point)
            if not improved:
                break
    
    return best_points
