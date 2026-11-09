"""Improved 3D container loading with 3D skyline lattice for O(1) collision checks."""
import time
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110  # safety margin under 120s
    
    # Generate all 6 rotations for each box
    def rotations(w, h, d):
        return list(set(itertools.permutations((w, h, d))))
    
    # Sort boxes by volume descending to place larger boxes first
    box_indices = list(range(len(boxes)))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in box_indices]
    sorted_indices = sorted(box_indices, key=lambda i: box_volumes[i], reverse=True)
    
    # 3D skyline lattice: skyline[x][y] = min z-coordinate of empty space above (x,y)
    # Initialize all to 0 (empty)
    skyline = np.zeros((W, H), dtype=int)
    
    placements = [None] * len(boxes)
    
    def can_place(x, y, z, w, h, d):
        """Check if a box can be placed at (x,y,z) with dimensions (w,h,d) using skyline."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # The box occupies x..x+w-1, y..y+h-1, z..z+d-1
        # For each cell in the footprint, the skyline must be <= z
        # (meaning the space from skyline[x,y] to H-1 is free, and z >= skyline[x,y])
        # Actually, skyline[x,y] represents the lowest z that is free above
        # So we need z >= skyline[x,y] for all cells in the footprint
        sub = skyline[x:x+w, y:y+h]
        if np.any(sub > z):
            return False
        return True
    
    def update_skyline(x, y, z, w, h, d):
        """Update skyline after placing a box at (x,y,z) with dims (w,h,d)."""
        # The box occupies z..z+d-1 in the height dimension
        # For each cell in the footprint, the new skyline is max(old, z+d)
        sub = skyline[x:x+w, y:y+h]
        new_val = z + d
        # Only update cells where the box actually blocks (i.e., where old skyline <= z)
        # But since we already verified can_place, all cells have skyline <= z
        # So we can simply set to max(current, z+d)
        np.maximum(sub, new_val, out=sub)
    
    def find_best_position(w, h, d):
        """Find the best position for a box using skyline-based O(1) check."""
        best_pos = None
        best_score = float('inf')
        
        # Iterate over all possible x, y positions
        for x in range(0, W - w + 1):
            for y in range(0, H - h + 1):
                # Get the maximum skyline value in the footprint
                sub = skyline[x:x+w, y:y+h]
                max_sky = np.max(sub) if sub.size > 0 else 0
                
                # The box must be placed at z >= max_sky
                z = max_sky
                if z + d > D:
                    continue
                
                # This position is valid
                score = z * W * H + y * W + x
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
        
        return best_pos
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        
        bw, bh, bd = boxes[idx]
        rots = rotations(bw, bh, bd)
        
        placed = False
        for rw, rh, rd in rots:
            pos = find_best_position(rw, rh, rd)
            if pos is not None:
                x, y, z = pos
                placements[idx] = (0, x, y, z, rw, rh, rd)
                update_skyline(x, y, z, rw, rh, rd)
                placed = True
                break
        
        if not placed:
            placements[idx] = (-1, 0, 0, 0, bw, bh, bd)
    
    # Fill any remaining None placements (shouldn't happen, but safety)
    for i in range(len(boxes)):
        if placements[i] is None:
            bw, bh, bd = boxes[i]
            placements[i] = (-1, 0, 0, 0, bw, bh, bd)
    
    return placements
