import time
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Preprocess boxes: Sort by volume descending (FFD)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Store placements in original index order
    # We will keep the best result across multiple strategy runs
    
    best_total_volume = -1
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # 2. Initialize Occupancy Grid
    max_grid_cells = 50_000_000
    use_dense_grid = (W * H * D) <= max_grid_cells
    
    # Define strategies: (sort_key_func, score_func)
    # sort_key_func determines the order in which EPs are checked
    # score_func determines the "best" among valid EPs (if not First Fit)
    
    # Common Sort Keys
    sort_zyx = lambda p: (p[2], p[1], p[0])
    sort_zxy = lambda p: (p[2], p[0], p[1])
    sort_yxz = lambda p: (p[1], p[0], p[2])
    sort_xyz = lambda p: (p[0], p[1], p[2])
    
    # Common Score Functions
    score_z = lambda p: p[2] * 1000 + p[1] * 100 + p[0]
    score_x = lambda p: p[0] * 1000 + p[1] * 100 + p[2]
    score_y = lambda p: p[1] * 1000 + p[0] * 100 + p[2]
    score_sum = lambda p: p[0] + p[1] + p[2]
    
    # Strategies to try
    # (sort_key, score_func)
    # If score_func is None, it's First Fit (take first valid in sorted order)
    strategies = [
        (sort_zyx, score_z),    # Bottom-Left-Front, Best Fit by Z
        (sort_zyx, score_sum),  # Bottom-Left-Front, Best Fit by Sum
        (sort_zxy, score_z),    # Bottom-Front-Left, Best Fit by Z
        (sort_yxz, score_y),    # Left-Front-Bottom, Best Fit by Y
        (sort_xyz, score_x),    # Front-Left-Bottom, Best Fit by X
        (sort_zyx, None),       # Bottom-Left-Front, First Fit
    ]
    
    for sort_key, score_func in strategies:
        # Check time
        if time.time() > deadline - 2.0:
            break
            
        # Initialize state for this attempt
        if use_dense_grid:
            grid = np.zeros((W, H, D), dtype=np.int8)
        else:
            grid = None
            
        extreme_points = [(0, 0, 0)]
        placed_boxes = []
        placements = [None] * n
        current_volume = 0
        
        # Iterate through sorted boxes
        for idx, w, h, d in indexed_boxes:
            if time.time() > deadline - 2.0:
                break
                
            # Get unique orientations for this box
            orients = list(set(itertools.permutations([w, h, d])))
            
            best_placement = None
            best_score_val = float('inf')
            
            # Sort extreme points
            sorted_eps = sorted(extreme_points, key=sort_key)
            
            for px, py, pz in sorted_eps:
                for ow, oh, od in orients:
                    # Check bounds
                    if px + ow > W or py + oh > H or pz + od > D:
                        continue
                    
                    # Check overlap
                    valid = True
                    if use_dense_grid:
                        if np.any(grid[px:px+ow, py:py+oh, pz:pz+od]):
                            valid = False
                    else:
                        for bx, by, bz, bw, bh, bd in placed_boxes:
                            if (px < bx + bw and bx < px + ow and
                                py < by + bh and by < py + oh and
                                pz < bz + bd and bz < pz + od):
                                valid = False
                                break
                    
                    if valid:
                        if score_func is None:
                            # First Fit
                            best_placement = (px, py, pz, ow, oh, od)
                            break # Break orientation loop
                        else:
                            # Best Fit
                            s = score_func((px, py, pz))
                            if s < best_score_val:
                                best_score_val = s
                                best_placement = (px, py, pz, ow, oh, od)
                
                if score_func is None and best_placement is not None:
                    break # Break EP loop if First Fit found a solution
            
            if best_placement is not None:
                px, py, pz, ow, oh, od = best_placement
                placements[idx] = (0, px, py, pz, ow, oh, od)
                placed_boxes.append((px, py, pz, ow, oh, od))
                current_volume += ow * oh * od
                
                if use_dense_grid:
                    grid[px:px+ow, py:py+oh, pz:pz+od] = 1
                
                # Generate new extreme points
                new_points = [
                    (px + ow, py, pz),
                    (px, py + oh, pz),
                    (px, py, pz + od)
                ]
                
                for p in new_points:
                    if p not in extreme_points:
                        extreme_points.append(p)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any remaining None
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        # Update best result
        if current_volume > best_total_volume:
            best_total_volume = current_volume
            best_placements = placements
    
    return best_placements
