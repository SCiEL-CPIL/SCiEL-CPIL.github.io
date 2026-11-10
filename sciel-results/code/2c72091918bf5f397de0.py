import time
import random
import numpy as np

def _score_fast(points):
    """
    Fast scoring using NumPy.
    points: list of tuples (x, y)
    Returns: max multiplicity of a non-zero doubled area.
    """
    P = np.array(points)
    n = len(P)
    
    # Max possible determinant for points in [-30, 30]
    # Max coordinate diff is 60.
    # Det = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    # Max absolute value roughly 30*60 + 30*60 + 30*60 = 5400.
    # Let's use a histogram size of 6000 to be safe.
    hist_size = 6000
    counts = np.zeros(hist_size, dtype=np.int32)
    
    for i in range(n):
        pi = P[i]
        idxs = np.arange(i + 1, n)
        m = len(idxs)
        if m < 2:
            continue
        
        Pj = P[idxs]
        
        # Generate indices for pairs (j, k) in idxs where j < k
        # np.triu_indices(m, k=1) returns (row, col) indices for upper triangle
        j_idx, k_idx = np.triu_indices(m, k=1)
        
        # v1 = P[j] - P[i]
        # v2 = P[k] - P[i]
        v1 = Pj[j_idx] - pi
        v2 = Pj[k_idx] - pi
        
        # Determinant
        dets = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        # Filter non-zero
        dets = dets[dets > 0]
        
        if len(dets) > 0:
            # Check bounds
            max_d = np.max(dets)
            if max_d < hist_size:
                # bincount returns array of length max_d+1
                # We need to add to counts
                # We can just use a temporary array and slice
                temp_counts = np.bincount(dets, minlength=hist_size)
                counts += temp_counts
            else:
                # If out of bounds, we might miss some counts, but unlikely
                # For safety, we could use a dictionary fallback, but for speed we ignore
                pass

    return int(np.max(counts))

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
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
                # Grid points:
                # x in [0, (W-1)*sx]
                # y in [0, (H-1)*sy]
                # We need to shift to fit in [-30, 30]
                # Center of grid: ((W-1)*sx)/2, ((H-1)*sy)/2
                # Shift dx, dy such that all points in [-30, 30]
                
                grid_width = (W - 1) * sx
                grid_height = (H - 1) * sy
                
                # Possible shifts
                # x_min = 0 + dx >= -30 => dx >= -30
                # x_max = grid_width + dx <= 30 => dx <= 30 - grid_width
                # So dx in [-30, 30 - grid_width]
                
                if 30 - grid_width < -30:
                    continue
                
                # We don't need to check all shifts, just a few representative ones
                # Because shifting doesn't change the relative areas, only the absolute coordinates.
                # Wait, the area is invariant under translation!
                # So the score is independent of dx, dy!
                # We only need to check if the grid FITS in the box.
                # If it fits, the score is the same for any valid shift.
                
                # So we just need to check if there EXISTS a shift.
                # If grid_width <= 60 and grid_height <= 60, it fits.
                
                if grid_width <= 60 and grid_height <= 60:
                    # Construct a valid grid (centered or at 0,0)
                    # Let's place it at 0,0 for simplicity, then shift if needed?
                    # No, just use 0,0 based if it fits in [-30, 30]?
                    # If W=30, sx=1, width=29. 0..29 fits in -30..30? Yes.
                    # If W=1, sx=1, width=0.
                    
                    # Generate points
                    points = [(x * sx, y * sy) for x in range(W) for y in range(H)]
                    
                    # Check if all points are in [-30, 30]
                    # Since we checked width/height <= 60, and coords are 0..W-1,
                    # max coord is (W-1)*sx <= 60.
                    # But min coord is 0.
                    # 0 is in [-30, 30].
                    # So yes, it fits.
                    
                    score = _score_fast(points)
                    if score > best_score:
                        best_score = score
                        best_points = points[:]
                    
                    # Also try swapped W, H?
                    # The factors list includes both (W,H) and (H,W) if they are different.
                    # But sx and sy are independent.
                    # So (W=6, H=5, sx=1, sy=1) is different from (W=5, H=6, sx=1, sy=1)
                    # But the set of points is the same shape, just rotated.
                    # The areas are the same.
                    # So we don't need to duplicate.
                    
    # 2. Local Search / Hill Climbing
    # Since _score_fast is fast, we can do more search.
    # But grids are likely optimal.
    # Let's try to perturb the best grid slightly?
    # Or just run a few random restarts with local search?
    
    if best_points is None:
        # Fallback
        best_points = [(i, 0) for i in range(30)]
        best_score = _score_fast(best_points)
        
    # Time check
    if time.time() > deadline - 5:
        return best_points
        
    # Local Search on the best grid
    # We can try moving one point at a time.
    # Since the grid is symmetric, moving a corner might be better?
    # Let's do a simple hill climbing.
    
    current_points = best_points[:]
    current_score = best_score
    n = 30
    
    # Pre-calculate all valid positions? No, just random nearby.
    # Or systematic: try moving each point to all nearby integer coords?
    # 30 points * 25 neighbors = 750 evals.
    # Each eval is ~1ms? 0.75s.
    # We can do this multiple times.
    
    for _ in range(10): # 10 rounds of hill climbing
        if time.time() > deadline - 2:
            break
            
        improved = False
        # Shuffle points to avoid bias? No, order doesn't matter for score.
        for idx in range(n):
            if time.time() > deadline - 2:
                break
                
            ox, oy = current_points[idx]
            
            # Try neighbors in a small window
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx = ox + dx
                    ny = oy + dy
                    
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    
                    # Check collision
                    if (nx, ny) in current_points:
                        continue
                    
                    # Apply move
                    current_points[idx] = (nx, ny)
                    new_score = _score_fast(current_points)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points[:]
                    else:
                        # Revert
                        current_points[idx] = (ox, oy)
                        
        if not improved:
            break
            
    return best_points
