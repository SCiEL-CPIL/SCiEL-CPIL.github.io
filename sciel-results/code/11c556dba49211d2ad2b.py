"""Skyline / floor-map based 3D container loading."""
import numpy as np
from itertools import product

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Use a 2D grid of minimum heights (skyline) for efficient overlap checks.
    # Grid dimensions: (W+1) x (D+1) if W, D are small enough.
    # If W or D is large, we need coordinate compression, but for typical container loading
    # problems with 60 boxes, W and D are likely manageable (e.g., < 1000).
    # However, to be safe and efficient, let's check if we can use a direct grid.
    # If W*D is too large (> 1M), we'll fall back to the list-based approach.
    # But for now, let's assume a direct grid is feasible for better performance.
    
    # Actually, let's use a coordinate compression approach to be robust.
    # We'll maintain sorted lists of x and z coordinates.
    # But updating a 2D grid with coordinate compression dynamically is complex.
    
    # Alternative: Since N=60 is small, let's stick with the list-based approach
    # but optimize it with numpy for the overlap check?
    # Or, let's try a simpler optimization: precompute the candidate positions
    # and use a more efficient data structure.
    
    # Let's try a different approach: maintain a 2D array of heights if W and D are small.
    # Check if W * D is reasonable.
    if W * D <= 1000000:  # 1M cells max
        # Use a 2D grid
        height_grid = np.zeros((W, D), dtype=np.int32)
        placed = []  # list of (x, y, z, w, h, d) for tracking
        
        box_indices = list(range(n))
        box_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        final_placements = [None] * n
        
        for idx in box_indices:
            w, h, d = boxes[idx]
            
            rotations = [
                (w, h, d),
                (w, d, h),
                (h, w, d),
                (h, d, w),
                (d, w, h),
                (d, h, w)
            ]
            
            best_placement = None
            best_y = float('inf')
            best_score = float('inf')  # Use a scoring function to prefer lower and more compact placements
            
            for rw, rh, rd in rotations:
                if rw > W or rh > H or rd > D:
                    continue
                
                # Generate candidate x and z positions
                # For a grid-based approach, we can scan all valid positions,
                # but that's O(W*D) which might be too slow.
                # Instead, use the same candidate generation as before:
                # x in {0} U {b.x + b.w}, z in {0} U {b.z + b.d}
                
                xs = {0}
                zs = {0}
                for (px, py, pz, pw, ph, pd) in placed:
                    xs.add(px + pw)
                    zs.add(pz + pd)
                
                xs = [x for x in xs if x + rw <= W]
                zs = [z for z in zs if z + rd <= D]
                
                for cx in xs:
                    for cz in zs:
                        # Use the grid to find the minimum y
                        # The box occupies x in [cx, cx+rw), z in [cz, cz+rd)
                        # We need the max height in this region
                        if cx + rw <= W and cz + rd <= D:
                            min_y = np.max(height_grid[cx:cx+rw, cz:cz+rd])
                            if min_y + rh <= H:
                                # Score: prefer lower y, and maybe prefer positions that fill gaps
                                score = min_y
                                if score < best_score:
                                    best_score = score
                                    best_y = min_y
                                    best_placement = (cx, min_y, cz, rw, rh, rd)
            
            if best_placement is not None:
                cx, cy, cz, rw, rh, rd = best_placement
                # Update the grid
                height_grid[cx:cx+rw, cz:cz+rd] = np.maximum(height_grid[cx:cx+rw, cz:cz+rd], cy + rh)
                placed.append((cx, cy, cz, rw, rh, rd))
                final_placements[idx] = (0, cx, cy, cz, rw, rh, rd)
            else:
                final_placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        return final_placements
    else:
        # Fallback to the original list-based approach for large containers
        placed = []
        box_indices = list(range(n))
        box_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        final_placements = [None] * n
        
        for idx in box_indices:
            w, h, d = boxes[idx]
            rotations = [
                (w, h, d),
                (w, d, h),
                (h, w, d),
                (h, d, w),
                (d, w, h),
                (d, h, w)
            ]
            
            best_placement = None
            best_y = float('inf')
            
            for rw, rh, rd in rotations:
                if rw > W or rh > H or rd > D:
                    continue
                
                xs = [0]
                zs = [0]
                for (px, py, pz, pw, ph, pd) in placed:
                    xs.append(px + pw)
                    zs.append(pz + pd)
                
                xs = [x for x in xs if x + rw <= W]
                zs = [z for z in zs if z + rd <= D]
                xs = sorted(set(xs))
                zs = sorted(set(zs))
                
                for cx in xs:
                    for cz in zs:
                        min_y = 0
                        for (px, py, pz, pw, ph, pd) in placed:
                            if cx < px + pw and cx + rw > px:
                                if cz < pz + pd and cz + rd > pz:
                                    if py + ph > min_y:
                                        min_y = py + ph
                        
                        if min_y + rh <= H:
                            if min_y < best_y:
                                best_y = min_y
                                best_placement = (cx, min_y, cz, rw, rh, rd)
            
            if best_placement is not None:
                cx, cy, cz, rw, rh, rd = best_placement
                placed.append((cx, cy, cz, rw, rh, rd))
                final_placements[idx] = (0, cx, cy, cz, rw, rh, rd)
            else:
                final_placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        return final_placements
