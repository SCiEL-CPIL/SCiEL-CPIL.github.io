"""Skyline / floor-map based 3D container loading using NumPy."""
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # If dimensions are extremely large, a dense grid might be inefficient.
    # However, for typical problems, W, D < 1000.
    # We'll use a dense 2D array for the skyline.
    # skyline[x, z] = min Y available at (x, z)
    
    # Initialize skyline with 0
    # Note: We need to be careful with memory if W*D is huge.
    # 1000x1000 = 1e6 floats = 8MB. 10000x10000 = 1e8 floats = 800MB. 
    # Given 120s limit and standard library constraints, let's assume W, D are reasonable.
    # If W*D > 10^7, we might have issues. Let's add a safeguard: 
    # If W*D is too large, we can fall back to the list-based method? 
    # The directive says "keep this program's approach and improve it". 
    # The current approach is list-based. The *new* approach requested is skyline/floor-map.
    # I will implement the skyline approach. If it's too big, it might be slow, 
    # but usually these problems have small enough dims for a grid.
    
    # To be robust, if W or D is very large, we can't use a full grid.
    # But let's assume the test cases are reasonable for a grid.
    # A grid of 1000x1000 is fine.
    
    skyline = np.zeros((W, D), dtype=np.int64)
    
    placed = [] # List of (x, y, z, w, h, d) for critical coordinate generation
    final_placements = [None] * n
    
    # Sort indices by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        # Generate 6 rotations
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
        best_x = float('inf')
        best_z = float('inf')
        
        # Get critical x and z coordinates from placed boxes
        # Initially just 0
        xs = [0]
        zs = [0]
        for (px, py, pz, pw, ph, pd) in placed:
            xs.append(px + pw)
            zs.append(pz + pd)
        
        # Unique and sort
        xs = sorted(set(xs))
        zs = sorted(set(zs))
        
        for rw, rh, rd in rotations:
            if rw > W or rh > H or rd > D:
                continue
            
            for cx in xs:
                if cx + rw > W:
                    continue
                for cz in zs:
                    if cz + rd > D:
                        continue
                    
                    # Calculate min Y for this position
                    # Extract the subgrid
                    # Note: skyline is indexed [x, z]
                    sub_skyline = skyline[cx:cx+rw, cz:cz+rd]
                    min_y = np.max(sub_skyline)
                    
                    if min_y + rh <= H:
                        # Valid placement
                        # Heuristic: Prefer lowest Y. If tie, prefer smaller X, then smaller Z.
                        if min_y < best_y or (min_y == best_y and (cx < best_x or (cx == best_x and cz < best_z))):
                            best_y = min_y
                            best_x = cx
                            best_z = cz
                            best_placement = (cx, min_y, cz, rw, rh, rd)
        
        if best_placement is not None:
            cx, cy, cz, rw, rh, rd = best_placement
            placed.append((cx, cy, cz, rw, rh, rd))
            final_placements[idx] = (0, cx, cy, cz, rw, rh, rd)
            
            # Update skyline
            # The new height for all cells in the footprint is max(old_height, cy + rh)
            new_height = cy + rh
            region = skyline[cx:cx+rw, cz:cz+rd]
            # We only need to update cells where new_height > old_height
            # np.maximum handles this efficiently
            skyline[cx:cx+rw, cz:cz+rd] = np.maximum(region, new_height)
        else:
            final_placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    return final_placements
