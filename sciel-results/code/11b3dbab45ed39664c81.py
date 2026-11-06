"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _orientations(w, h, d):
    """Return all 6 unique axis-aligned rotations of a box (w,h,d)."""
    s = {(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)}
    return list(s)

def _overlaps(x, y, z, w, h, d, placed):
    """Check if box at (x,y,z) with size (w,h,d) overlaps any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if x < px + pw and x + w > px and \
           y < py + pd and y + h > py and \
           z < pz + ph and z + d > pz:
            return True
    return False

def _fits(x, y, z, w, h, d, W, H, D):
    """Check if box fits within container at position (x,y,z)."""
    return x >= 0 and y >= 0 and z >= 0 and \
           x + w <= W and y + h <= H and z + d <= D

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda x: (-(x[1][0] * x[1][1] * x[1][2]),
                                 -max(x[1])))
    
    # placements[i] = (x, y, z, w, h, d) for box i, or None if not placed
    placed_list = []  # list of (x, y, z, w, h, d) for quick overlap check
    result = [None] * n  # will hold (bin, x, y, z, w, h, d)
    
    # Track which original index each placed box corresponds to
    placed_info = []  # list of (orig_idx, x, y, z, w, h, d)
    
    for orig_idx, (w, h, d) in indexed:
        if time.time() > deadline:
            break
        
        best_score = None
        best_pos = None
        best_orient = None
        
        # Generate candidate points
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed_list:
            # On top
            candidates.add((px, py, pz + ph))
            # Beside in x
            candidates.add((px + pw, py, pz))
            # Beside in y
            candidates.add((px, py + pd, pz))
            # Corners: combinations
            candidates.add((px + pw, py, pz + ph))
            candidates.add((px, py + pd, pz + ph))
            candidates.add((px + pw, py + pd, pz))
            candidates.add((px + pw, py + pd, pz + ph))
            # Also try placing at the "corner" formed by two boxes
            # For efficiency, limit candidate count
            if len(candidates) > 200:
                break
        
        # Try all orientations and candidate positions
        orients = _orientations(w, h, d)
        
        for ow, oh, od in orients:
            for (cx, cy, cz) in candidates:
                if not _fits(cx, cy, cz, ow, oh, od, W, H, D):
                    continue
                if _overlaps(cx, cy, cz, ow, oh, od, placed_list):
                    continue
                
                # Score: prefer positions closer to origin (best-fit)
                # Use a weighted score to encourage compact packing
                score = cx * 1.0 + cy * 1.0 + cz * 1.0
                # Also penalize if it extends far in any dimension
                score += max(0, cx + ow - W) * 10  # shouldn't happen if fits
                score += max(0, cy + oh - H) * 10
                score += max(0, cz + od - D) * 10
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_orient = (ow, oh, od)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_orient
            placed_list.append((bx, by, bz, bw, bh, bd))
            placed_info.append((orig_idx, bx, by, bz, bw, bh, bd))
            result[orig_idx] = (0, bx, by, bz, bw, bh, bd)
    
    # Fill in unplaced boxes
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
