"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _get_all_orientations(w, h, d):
    """Return unique orientations of a box (w,h,d)."""
    seen = set()
    orientations = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            orientations.append(perm)
    return orientations

def _fits(placed, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _candidate_points(placed, W, H, D):
    """Generate candidate (x,y,z) positions to try for placement."""
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        # Points adjacent to each face of each placed box
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
        candidates.append((px, py, pz + pd))
    return candidates

def _skyline_solve(boxes, container, time_limit=10.0):
    """
    Skyline-based heuristic: maintains a 2D grid of minimum Z-heights per (x,y) cell.
    Uses discrete grid with cell size 1 for fast collision checks.
    Returns list of placements: (bin, x, y, z, w, h, d) per box.
    """
    W, H, D = container
    container_vol = W * H * D
    
    # Precompute all orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(_get_all_orientations(w, h, d))
    
    # Sorting strategies
    def sort_by_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_footprint_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(w*h for w,h,d in box_orientations[i]), reverse=True)
    
    def sort_by_height_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(d for w,h,d in box_orientations[i]), reverse=True)
    
    def sort_by_volume_asc(idx_list):
        return sorted(idx_list, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    strategies = [
        sort_by_volume_desc,
        sort_by_max_dim_desc,
        sort_by_footprint_desc,
        sort_by_height_desc,
        sort_by_volume_asc,
    ]
    
    all_indices = list(range(len(boxes)))
    best_placements = None
    best_score = 0.0
    
    start_time = time.time()
    
    for strategy in strategies:
        if time.time() - start_time > time_limit:
            break
        
        order = strategy(all_indices)
        
        # Skyline: 2D array where skyline[x][y] = min z-height above that cell
        # Initialize all to 0
        skyline = [[0] * (H + 1) for _ in range(W + 1)]
        
        placed = []  # List of (x, y, z, w, h, d)
        box_placed = [False] * len(boxes)
        box_pos = [None] * len(boxes)
        
        for idx in order:
            w, h, d = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_score_pos = None
            
            # Generate candidate x, y positions
            # For skyline approach, we try all possible (x, y) positions
            # But that's O(W*H) which can be large. Let's use a smarter approach:
            # Only try positions where x and y are at boundaries of existing boxes or 0
            
            candidate_xs = set([0])
            candidate_ys = set([0])
            for (px, py, pz, pw, ph, pd) in placed:
                candidate_xs.add(px)
                candidate_xs.add(px + pw)
                candidate_ys.add(py)
                candidate_ys.add(py + ph)
            
            # Filter candidates that fit
            candidate_xs = [x for x in candidate_xs if x <= W]
            candidate_ys = [y for y in candidate_ys if y <= H]
            
            # Sort candidates to prefer lower positions
            candidate_xs.sort()
            candidate_ys.sort()
            
            found = False
            for ow, oh, od in orientations:
                for cx in candidate_xs:
                    if cx + ow > W:
                        continue
                    for cy in candidate_ys:
                        if cy + oh > H:
                            continue
                        
                        # Check skyline: find max z in the footprint region
                        max_z = 0
                        for gx in range(cx, cx + ow):
                            for gy in range(cy, cy + oh):
                                if skyline[gx][gy] > max_z:
                                    max_z = skyline[gx][gy]
                        
                        cz = max_z
                        if cz + od > D:
                            continue
                        
                        # Verify no overlap using placed list (safety check)
                        if not _fits(placed, cx, cy, cz, ow, oh, od):
                            continue
                        
                        # Score: prefer lower z, then lower y, then lower x
                        pos_score = (cz, cy, cx)
                        if best_pos is None or pos_score < best_score_pos:
                            best_pos = (cx, cy, cz, ow, oh, od)
                            best_score_pos = pos_score
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            
            if best_pos is not None:
                cx, cy, cz, ow, oh, od = best_pos
                placed.append(best_pos)
                box_placed[idx] = True
                box_pos[idx] = best_pos
                
                # Update skyline
                for gx in range(cx, cx + ow):
                    for gy in range(cy, cy + oh):
                        new_z = cz + od
                        if new_z > skyline[gx][gy]:
                            skyline[gx][gy] = new_z
        
        # Calculate score
        total_vol = sum(p[3]*p[4]*p[5] for p in placed)
        score = total_vol / container_vol
        
        if score > best_score:
            best_score = score
            result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
            for i in range(len(boxes)):
                if box_placed[i]:
                    x, y, z, w, h, d = box_pos[i]
                    result[i] = (0, x, y, z, w, h, d)
            best_placements = result
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
    
    return best_placements

def _blb_solve(boxes, container, time_limit=10.0):
    """
    Bottom-Left-Back heuristic with multiple sorting strategies and restarts.
    Returns list of placements: (bin, x, y, z, w, h, d) per box.
    """
    W, H, D = container
    container_vol = W * H * D
    
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(_get_all_orientations(w, h, d))
    
    def sort_by_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_footprint_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(w*h for w,h,d in box_orientations[i]), reverse=True)
    
    def sort_by_height_desc(idx_list):
        return sorted(idx_list, key=lambda i: max(d for w,h,d in box_orientations[i]), reverse=True)
    
    def sort_by_volume_asc(idx_list):
        return sorted(idx_list, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    strategies = [
        sort_by_volume_desc,
        sort_by_max_dim_desc,
        sort_by_footprint_desc,
        sort_by_height_desc,
        sort_by_volume_asc,
    ]
    
    all_indices = list(range(len(boxes)))
    best_placements = None
    best_score = 0.0
    
    start_time = time.time()
    
    for strategy in strategies:
        if time.time() - start_time > time_limit:
            break
        
        order = strategy(all_indices)
        placed = []
        box_placed = [False] * len(boxes)
        box_pos = [None] * len(boxes)
        
        for idx in order:
            w, h, d = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_score_pos = None
            
            candidates = _candidate_points(placed, W, H, D)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
            
            found = False
            for ow, oh, od in orientations:
                for (cx, cy, cz) in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if _fits(placed, cx, cy, cz, ow, oh, od):
                        pos_score = (cz, cy, cx)
                        if best_pos is None or pos_score < best_score_pos:
                            best_pos = (cx, cy, cz, ow, oh, od)
                            best_score_pos = pos_score
                        found = True
                        break
                if found:
                    break
            
            if best_pos is not None:
                placed.append(best_pos)
                box_placed[idx] = True
                box_pos[idx] = best_pos
        
        total_vol = sum(p[3]*p[4]*p[5] for p in placed)
        score = total_vol / container_vol
        
        if score > best_score:
            best_score = score
            result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
            for i in range(len(boxes)):
                if box_placed[i]:
                    x, y, z, w, h, d = box_pos[i]
                    result[i] = (0, x, y, z, w, h, d)
            best_placements = result
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
    
    return best_placements

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    any_fits = False
    for w, h, d in boxes:
        for ow, oh, od in _get_all_orientations(w, h, d):
            if ow <= W and oh <= H and od <= D:
                any_fits = True
                break
        if any_fits:
            break
    
    if not any_fits:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start = time.time()
    deadline = 110.0
    
    # Try both skyline and BLB solvers, keep the best result
    results = []
    
    # Run skyline solver first (often faster for dense packings)
    remaining = deadline - (time.time() - start) - 10.0
    if remaining > 5.0:
        sky_result = _skyline_solve(boxes, container, time_limit=remaining)
        results.append(sky_result)
    
    # Run BLB solver with remaining time
    remaining = deadline - (time.time() - start) - 5.0
    if remaining > 5.0:
        blb_result = _blb_solve(boxes, container, time_limit=remaining)
        results.append(blb_result)
    
    # Pick the best result
    if not results:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_vol = W * H * D
    best_result = results[0]
    best_score = 0.0
    
    for res in results:
        total_vol = sum(x[4]*x[5]*x[6] for x in res if x[0] == 0)
        score = total_vol / container_vol
        if score > best_score:
            best_score = score
            best_result = res
    
    return best_result
# EVOLVE-BLOCK-END
