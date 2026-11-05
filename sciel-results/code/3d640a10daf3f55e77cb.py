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
    
    # Sort boxes by volume descending
    box_volumes = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    
    # Placement data structure
    placed = []  # list of tuples (x, y, z, w, h, d)
    
    # Candidate points for placement
    candidate_points = [(0, 0, 0)]
    
    def can_place(px, py, pz, pw, ph, pd):
        if px < 0 or py < 0 or pz < 0:
            return False
        if px + pw > W or py + ph > H or pz + pd > D:
            return False
        
        for (ex, ey, ez, ew, eh, ed) in placed:
            if (px < ex + ew and ex < px + pw and
                py < ey + eh and ey < py + ph and
                pz < ez + ed and ez < pz + pd):
                return False
        return True
    
    placements = [None] * n
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        for (rw, rh, rd) in box_rotations[idx]:
            for (px, py, pz) in candidate_points:
                if can_place(px, py, pz, rw, rh, rd):
                    # Heuristic: Minimize z, then y, then x
                    score = pz * 1000000 + py * 1000 + px
                    if score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            px, py, pz = best_pos
            rw, rh, rd = best_rot
            placements[idx] = (0, px, py, pz, rw, rh, rd)
            placed.append((px, py, pz, rw, rh, rd))
            
            # Generate new candidate points (outer corners)
            new_points = [
                (px + rw, py, pz),
                (px, py + rh, pz),
                (px, py, pz + rd),
                (px + rw, py + rh, pz),
                (px + rw, py, pz + rd),
                (px, py + rh, pz + rd),
                (px + rw, py + rh, pz + rd)
            ]
            candidate_points.extend(new_points)
            
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
