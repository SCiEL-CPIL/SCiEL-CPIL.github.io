"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate all 6 possible rotations for each box
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in permutations([w, h, d]):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    # Precompute rotations for all boxes
    box_rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Helper to check if a box fits in the container
    def fits_in_container(px, py, pz, pw, ph, pd):
        return (px >= 0 and py >= 0 and pz >= 0 and 
                px + pw <= W and py + ph <= H and pz + pd <= D)

    # Helper to check overlap
    def has_overlap(px, py, pz, pw, ph, pd, placed_boxes):
        for (ex, ey, ez, ew, eh, ed) in placed_boxes:
            if (px < ex + ew and ex < px + pw and
                py < ey + eh and ey < py + ph and
                pz < ez + ed and ez < pz + pd):
                return True
        return False

    # Strategy 1: Volume Descending
    # Strategy 2: Max Dimension Descending
    # Strategy 3: "Compactness" (Volume / Max Edge) Descending
    # Strategy 4: Sum of Dimensions Descending
    
    strategies = [
        sorted(range(n), key=lambda i: box_volumes[i], reverse=True),
        sorted(range(n), key=lambda i: max(boxes[i]), reverse=True),
        sorted(range(n), key=lambda i: box_volumes[i] / max(boxes[i]) if max(boxes[i]) > 0 else 0, reverse=True),
        sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True),
    ]
    
    best_total_volume = 0
    best_placements = None
    
    for order in strategies:
        placements = [None] * n
        placed_boxes = []
        total_vol = 0
        
        # Track bounding box of placed items to use for scoring
        # Initially, bounding box is 0,0,0 to 0,0,0
        min_x, min_y, min_z = 0, 0, 0
        max_x, max_y, max_z = 0, 0, 0
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for (rw, rh, rd) in box_rotations[idx]:
                # Candidate points: 
                # We need a set of points to check. 
                # A robust set includes (0,0,0) and the "faces" of existing boxes.
                # To keep it fast, we use a simplified candidate generation:
                # Start with (0,0,0). For each placed box, add its 8 corners.
                # This is O(N^2 * R * C) which might be slow for 60 boxes if N is large,
                # but 60 is small. 60^2 * 6 * 60 ~ 1.3M checks. This is fine in Python if optimized.
                
                # Optimization: Pre-generate candidate points once per step? No, they grow.
                # We will generate a set of unique candidate points.
                candidates = set()
                candidates.add((0, 0, 0))
                for (ex, ey, ez, ew, eh, ed) in placed_boxes:
                    # Add corners
                    for dx in [0, ew]:
                        for dy in [0, eh]:
                            for dz in [0, ed]:
                                candidates.add((ex + dx, ey + dy, ez + dz))
                
                for (px, py, pz) in candidates:
                    if not fits_in_container(px, py, pz, rw, rh, rd):
                        continue
                    if has_overlap(px, py, pz, rw, rh, rd, placed_boxes):
                        continue
                    
                    # Calculate Score
                    # Heuristic: Minimize the volume of the bounding box of all placed + this box
                    new_min_x = min(min_x, px)
                    new_min_y = min(min_y, py)
                    new_min_z = min(min_z, pz)
                    new_max_x = max(max_x, px + rw)
                    new_max_y = max(max_y, py + rh)
                    new_max_z = max(max_z, pz + rd)
                    
                    # Since we start at 0,0,0 and only place in positive octant, min is always 0
                    # So we just maximize the spread? No, we want to MINIMIZE the spread.
                    # Bounding Box Volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
                    # We want to minimize this.
                    bb_vol = (new_max_x) * (new_max_y) * (new_max_z)
                    
                    # Tie-breaker: Prefer lower Z, then Y, then X
                    score = bb_vol * 100000 + pz * 1000 + py * 100 + px
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                px, py, pz = best_pos
                rw, rh, rd = best_rot
                placements[idx] = (0, px, py, pz, rw, rh, rd)
                placed_boxes.append((px, py, pz, rw, rh, rd))
                total_vol += box_volumes[idx]
                
                # Update bounding box
                max_x = max(max_x, px + rw)
                max_y = max(max_y, py + rh)
                max_z = max(max_z, pz + rd)
                # mins are always 0 in this setup
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        if total_vol > best_total_volume:
            best_total_volume = total_vol
            best_placements = placements[:]
            
    # Fill any remaining None with -1
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    else:
        for i in range(n):
            if best_placements[i] is None:
                w, h, d = boxes[i]
                best_placements[i] = (-1, 0, 0, 0, w, h, d)
                
    return best_placements
# EVOLVE-BLOCK-END
