import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # bins: list of lists of (x, y, z, w, h, d, orig_idx)
    bins = []
    bin_volumes = []  # cumulative volume in each bin
    # Occupancy grids for each bin (numpy boolean arrays)
    bin_grids = []
    
    # Helper: create an empty grid for a bin
    def make_grid():
        return np.zeros((W, H, D), dtype=np.bool_)
    
    # Helper: check if a box at (cx, cy, cz) with dims (w, h, d) fits without overlap
    def check_fit(grid, cx, cy, cz, w, h, d):
        if cx + w > W or cy + h > H or cz + d > D:
            return False
        if w == 0 or h == 0 or d == 0:
            return True
        return not np.any(grid[cx:cx+w, cy:cy+h, cz:cz+d])
    
    # Helper: place a box on the grid
    def place_on_grid(grid, cx, cy, cz, w, h, d):
        grid[cx:cx+w, cy:cy+h, cz:cz+d] = True
    
    # Helper: generate candidate positions for a bin
    def generate_candidates(bin_boxes, w, h, d):
        candidates = set()
        candidates.add((0, 0, 0))
        
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            # Face points
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            
            # Combinations of face points for better gap-filling
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
        
        # Filter out positions that exceed container bounds
        valid_candidates = []
        for cx, cy, cz in candidates:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                valid_candidates.append((cx, cy, cz))
        
        return valid_candidates
    
    # Helper: score a placement (lower is better)
    def placement_score(cx, cy, cz, w, h, d):
        # Minimize distance to origin (packs tighter toward corner)
        # Use weighted sum to prefer packing in a specific order
        return cx + cy + cz
    
    # Main packing loop
    for w, h, d, orig_idx in indexed_boxes:
        # Check time
        if time.time() > deadline:
            # Fallback: place remaining boxes in new bins at (0,0,0)
            for rem_w, rem_h, rem_d, rem_idx in indexed_boxes:
                if rem_idx == orig_idx:
                    continue
                bins.append([])
                grid = make_grid()
                place_on_grid(grid, 0, 0, 0, rem_w, rem_h, rem_d)
                bins[-1].append((0, 0, 0, rem_w, rem_h, rem_d, rem_idx))
                bin_volumes.append(rem_w * rem_h * rem_d)
                bin_grids.append(grid)
            # Also place the current box
            bins.append([])
            grid = make_grid()
            place_on_grid(grid, 0, 0, 0, w, h, d)
            bins[-1].append((0, 0, 0, w, h, d, orig_idx))
            bin_volumes.append(w * h * d)
            bin_grids.append(grid)
            break
        
        best_bin_idx = -1
        best_pos = None
        best_score = None
        
        # Try existing bins, preferring fuller bins first (to minimize B)
        if bins:
            bin_order = sorted(range(len(bins)), key=lambda i: bin_volumes[i], reverse=True)
        else:
            bin_order = []
        
        for b_idx in bin_order:
            bin_boxes = bins[b_idx]
            grid = bin_grids[b_idx]
            
            # Generate candidate positions
            candidates = generate_candidates(bin_boxes, w, h, d)
            
            for cx, cy, cz in candidates:
                # Check fit using grid
                if not check_fit(grid, cx, cy, cz, w, h, d):
                    continue
                
                # Score
                score = placement_score(cx, cy, cz, w, h, d)
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin_idx = b_idx
                    best_pos = (cx, cy, cz)
        
        if best_bin_idx == -1:
            # Open new bin
            bins.append([])
            bin_volumes.append(0)
            grid = make_grid()
            bin_grids.append(grid)
            best_bin_idx = len(bins) - 1
            best_pos = (0, 0, 0)
        
        # Place the box
        bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_idx))
        bin_volumes[best_bin_idx] += w * h * d
        place_on_grid(bin_grids[best_bin_idx], best_pos[0], best_pos[1], best_pos[2], w, h, d)
    
    # Construct result in original order
    final_placements = [None] * len(boxes)
    for b_idx, bin_boxes in enumerate(bins):
        for x, y, z, w, h, d, orig_idx in bin_boxes:
            final_placements[orig_idx] = (b_idx, x, y, z, w, h, d)
    
    return final_placements
