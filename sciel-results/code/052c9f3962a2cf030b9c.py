"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _try_place(boxes, container, seed=0):
    """
    Attempt to place boxes into the container using an Extreme Point / skyline-like heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort boxes by volume descending to pack large items first
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # Placements: list of (x, y, z, w, h, d) for placed boxes
    placed = []
    # Result: for each original box index, store placement or None
    result = [None] * n
    
    # Candidate positions: start with (0,0,0)
    candidates = [(0, 0, 0)]
    
    for idx in box_indices:
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Try each candidate position
            for cx, cy, cz in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check overlap with all placed boxes
                overlap = False
                for px, py, pz, pw, ph, pd in placed:
                    if not (cx + rw <= px or px + pw <= cx or
                            cy + rh <= py or py + ph <= cy or
                            cz + rd <= pz or pz + pd <= cz):
                        overlap = True
                        break
                if overlap:
                    continue
                
                # Score: prefer positions closer to origin (lower x, y, z)
                score = cx * 1.0 + cy * 1.0 + cz * 1.0
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidate positions from this placement
            new_candidates = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd),
            ]
            for nc in new_candidates:
                # Only add if within container bounds
                if nc[0] <= W and nc[1] <= H and nc[2] <= D:
                    # Check if this candidate is not already dominated
                    # Simple check: is it inside any existing candidate's "box"?
                    # For efficiency, just add it; we'll deduplicate later if needed
                    candidates.append(nc)
            
            # Periodically clean up candidates to keep list manageable
            if len(candidates) > 500:
                # Keep only candidates that are not inside any placed box
                filtered = []
                for cx, cy, cz in candidates:
                    inside = False
                    for px, py, pz, pw, ph, pd in placed:
                        if px <= cx < px + pw and py <= cy < py + ph and pz <= cz < pz + pd:
                            inside = True
                            break
                    if not inside:
                        filtered.append((cx, cy, cz))
                # Deduplicate
                seen = set()
                unique = []
                for c in filtered:
                    if c not in seen:
                        seen.add(c)
                        unique.append(c)
                candidates = unique[:500]
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        # Check time
        if time.time() > deadline:
            # Fill remaining with -1
            for i in range(n):
                if result[i] is None:
                    w_i, h_i, d_i = boxes[i]
                    result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
            break
    
    # Fill any remaining None entries
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    
    # Try the main heuristic
    placements = _try_place(boxes, container)
    
    return placements
# EVOLVE-BLOCK-END
