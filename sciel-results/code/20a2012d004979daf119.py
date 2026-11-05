"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 120s limit, keep 5s margin for safety/overhead
    deadline = start_time + 115.0
    
    # Preprocess: calculate volume and sort by volume descending (Best-Fit Decreasing)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # Placements will be stored in original index order
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Current placed boxes: list of (x, y, z, w, h, d)
    placed = []
    
    def get_rotations(w, h, d):
        # Generate unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def find_best_position(bw, bh, bd):
        """
        Finds the best position (x, y, z) for a box of size (bw, bh, bd)
        using Bottom-Left-Far heuristic.
        Returns (x, y, z) or None if no position found.
        """
        # Candidate x-coordinates: 0 and right faces of all placed boxes
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
            
        # Filter candidates that fit within container bounds
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None

        best_pos = None
        # Heuristic: Minimize z (bottom), then x (left), then y (far/back)
        # Wait, standard BLF is Bottom-Left-Far.
        # Bottom: min z
        # Left: min x
        # Far: max y (to pack against the back wall)
        # Let's check standard definitions.
        # Usually "Far" means pushing away from the "front" (y=0) towards the "back" (y=H).
        # So we want to maximize y.
        
        best_score = None
        best_coord = None
        
        for z in sorted(z_candidates):
            for x in sorted(x_candidates):
                # For a given z and x, we want the largest y that doesn't overlap
                # Actually, it's easier to iterate all candidates and check overlap, then pick best.
                # With 60 boxes, N^3 is 216,000 checks per box. 60*216,000 = 13M ops.
                # Python might be slow for 13M ops in 115s?
                # 13M simple ops in Python is ~1-2 seconds. So it's fine.
                
                for y in y_candidates:
                    # Check overlap
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (x + bw <= px or px + pw <= x or 
                                y + bh <= py or py + ph <= y or 
                                z + bd <= pz or pz + pd <= z):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Score: Minimize z, then Minimize x, then Maximize y
                        # We want the "lowest" z, then "leftmost" x, then "farthest" y
                        # Score tuple: (z, x, -y) -> min
                        score = (z, x, -y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            
        return best_coord

    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        rots = get_rotations(w, h, d)
        placed_this = False
        
        for rw, rh, rd in rots:
            pos = find_best_position(rw, rh, rd)
            if pos is not None:
                x, y, z = pos
                placed.append((x, y, z, rw, rh, rd))
                final_placements[idx] = (0, x, y, z, rw, rh, rd)
                placed_this = True
                break
                
    return final_placements
# EVOLVE-BLOCK-END
