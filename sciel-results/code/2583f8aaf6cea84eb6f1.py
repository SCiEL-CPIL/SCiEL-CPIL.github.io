"""Optimized Skyline / floor-map based 3D container loading using NumPy."""
import numpy as np
from itertools import product

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Skyline grid: skyline[x, z] = max height used at (x, z)
    # This represents the "floor" - the lowest available y at each (x,z) position
    skyline = np.zeros((W, D), dtype=np.int64)
    
    # Track placed boxes for coordinate generation
    placed_boxes = []  # List of (x, y, z, w, h, d)
    final_placements = [None] * n
    
    # Sort by volume descending, then by max dimension descending for better packing
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], 
                                     max(boxes[i][0], boxes[i][1], boxes[i][2])), 
                     reverse=True)
    
    # Cache for critical coordinates to avoid recomputing
    x_coords = {0}
    z_coords = {0}
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        # Generate unique rotations (avoid duplicates)
        rotations = set()
        for rw, rh, rd in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if rw <= W and rh <= H and rd <= D:
                rotations.add((rw, rh, rd))
        
        best_placement = None
        best_score = float('inf')
        
        # Build candidate x and z positions from existing placed boxes
        # Include 0 and all edges of placed boxes
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            x_coords.add(px)
            x_coords.add(px + pw)
            z_coords.add(pz)
            z_coords.add(pz + pd)
        
        # Filter to valid positions
        valid_xs = sorted([x for x in x_coords if 0 <= x <= W])
        valid_zs = sorted([z for z in z_coords if 0 <= z <= D])
        
        for rw, rh, rd in rotations:
            for cx in valid_xs:
                if cx + rw > W:
                    continue
                for cz in valid_zs:
                    if cz + rd > D:
                        continue
                    
                    # Calculate the maximum height in the footprint region
                    # This is the lowest valid y position for this box
                    region = skyline[cx:cx+rw, cz:cz+rd]
                    min_y = int(np.max(region)) if region.size > 0 else 0
                    
                    if min_y + rh > H:
                        continue
                    
                    # Score: prefer lower y, then smaller x, then smaller z
                    # Also prefer placements that are more "compact" (closer to origin)
                    score = min_y * 1000000 + cx * 1000 + cz
                    
                    if score < best_score:
                        best_score = score
                        best_placement = (cx, min_y, cz, rw, rh, rd)
        
        if best_placement is not None:
            cx, cy, cz, rw, rh, rd = best_placement
            placed_boxes.append((cx, cy, cz, rw, rh, rd))
            final_placements[idx] = (0, cx, cy, cz, rw, rh, rd)
            
            # Update skyline: set all cells in footprint to at least cy + rh
            new_height = cy + rh
            skyline[cx:cx+rw, cz:cz+rd] = np.maximum(skyline[cx:cx+rw, cz:cz+rd], new_height)
        else:
            final_placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    return final_placements
