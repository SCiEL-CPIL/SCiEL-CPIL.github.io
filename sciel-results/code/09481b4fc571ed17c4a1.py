"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _check_overlap(cx, cy, cz, rw, rh, rd, placed):
    """Check if a box at (cx,cy,cz) with size (rw,rh,rd) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (cx + rw <= px or px + pw <= cx or
                cy + rh <= py or py + ph <= cy or
                cz + rd <= pz or pz + pd <= cz):
            return True
    return False

def _generate_candidates(placed, W, H, D, max_candidates=400):
    """
    Generate candidate positions from placed boxes using extreme points.
    Returns a list of candidate (x, y, z) positions.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    for px, py, pz, pw, ph, pd in placed:
        # New extreme points from each placed box
        new_pts = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for pt in new_pts:
            if 0 <= pt[0] <= W and 0 <= pt[1] <= H and 0 <= pt[2] <= D:
                candidates.add(pt)
    
    # Convert to list and limit size
    cand_list = list(candidates)
    if len(cand_list) > max_candidates:
        # Sort by priority (closer to origin first)
        cand_list.sort(key=lambda c: c[0] + c[1] + c[2])
        cand_list = cand_list[:max_candidates]
    
    return cand_list

def _try_place(boxes, container, seed=0):
    """
    Attempt to place boxes into the container using an Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
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
    
    for idx in box_indices:
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        # Generate candidate positions
        candidates = _generate_candidates(placed, W, H, D)
        
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
                if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
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

def _try_place_diverse(boxes, container, seed=0):
    """
    Try multiple sorting strategies and return the best result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # Define different sorting strategies
    def sort_by_volume_desc(indices):
        return sorted(indices, key=lambda i: box_volumes[i], reverse=True)
    
    def sort_by_aspect_ratio(indices):
        # Sort by max/min dimension ratio (more square first)
        def aspect(i):
            w, h, d = boxes[i]
            dims = sorted([w, h, d])
            return dims[2] / max(dims[0], 1)
        return sorted(indices, key=aspect)
    
    def sort_by_surface_area(indices):
        def surf(i):
            w, h, d = boxes[i]
            return 2 * (w*h + h*d + w*d)
        return sorted(indices, key=surf, reverse=True)
    
    def sort_by_max_dim(indices):
        def maxd(i):
            w, h, d = boxes[i]
            return max(w, h, d)
        return sorted(indices, key=maxd, reverse=True)
    
    strategies = [sort_by_volume_desc, sort_by_aspect_ratio, sort_by_surface_area, sort_by_max_dim]
    
    best_result = None
    best_filled = 0
    
    for strat_idx, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        # Create a copy of boxes for this strategy
        box_indices = list(range(n))
        box_indices = strategy(box_indices)
        
        # Place boxes using the extreme point heuristic
        placed = []
        result = [None] * n
        
        for idx in box_indices:
            if time.time() > deadline:
                break
            
            w_orig, h_orig, d_orig = boxes[idx]
            rots = box_rotations[idx]
            
            # Generate candidate positions
            candidates = _generate_candidates(placed, W, H, D)
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots:
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                        continue
                    
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
            else:
                result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        # Fill any remaining None entries
        for i in range(n):
            if result[i] is None:
                w_i, h_i, d_i = boxes[i]
                result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
        
        # Calculate filled volume
        filled = 0
        for r in result:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        
        if filled > best_filled:
            best_filled = filled
            best_result = result
    
    if best_result is None:
        # Fallback: all boxes left out
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_result

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    
    # Try the diverse strategy which explores multiple sorting orders
    placements = _try_place_diverse(boxes, container)
    
    return placements
# EVOLVE-BLOCK-END
