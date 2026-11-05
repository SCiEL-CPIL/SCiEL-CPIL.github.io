"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _try_place(boxes, container, sort_key=None):
    """
    Attempt to place boxes into the container using Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Sort rotations: prefer lower height (z-dim) to stack better, then lower depth, then lower width
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    # Sort boxes based on sort_key
    box_indices = list(range(n))
    if sort_key is None:
        box_volumes = [w * h * d for w, h, d in boxes]
        box_indices.sort(key=lambda i: -box_volumes[i])
    else:
        box_indices.sort(key=sort_key)
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Candidate points for placement (Extreme Points)
    # Each point is (x, y, z). We try to place boxes at these points.
    # Simplified: Start with (0,0,0). When a box is placed at (x,y,z) with dims (w,h,d),
    # add (x+w, y, z), (x, y+h, z), (x, y, z+d) to candidates if they are within bounds and not inside existing boxes.
    candidates = [(0, 0, 0)]
    
    def is_inside_existing(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if px >= ex and px + pw <= ex + ew and \
               py >= ey and py + ph <= ey + eh and \
               pz >= ez and pz + pd <= ez + ed:
                return True
        return False

    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            # Check if this point is strictly inside an existing box (if so, it's not a surface point)
            # Actually, for extreme point, we just need to check if a box *starting* here overlaps.
            # We add it to the list, and we will filter/sort later or check validity during placement.
            # To keep the list small, we can check if it's covered by existing boxes.
            # A point (x,y,z) is covered if there is a box containing it.
            # But we place boxes *at* the point, so the point is the bottom-left-back corner.
            # We should only keep points that are not inside any existing box.
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Sort candidates by Z (lowest), then Y (lowest), then X (leftmost)
        # This promotes compact packing towards the origin
        # We need to check all candidates, but try the "best" ones first
        # To be safe and fast, we iterate through a sorted copy of candidates
        # Note: candidates list grows. We should probably limit its size or clean it up.
        # For 60 boxes, the number of extreme points is manageable.
        
        # Filter candidates: only keep those that are within container bounds
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        # Sort them
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        # To improve performance, we can limit the number of candidates we try per box
        # If we have too many, we might miss a spot. But usually, the "lowest" spots are best.
        # Let's try all valid candidates, but break early if found.
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            # Try each candidate point
            found = False
            # We iterate through sorted candidates
            for cx, cy, cz in valid_candidates:
                # Check bounds
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check overlap
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    # Place box
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    # Generate new candidates
                    add_candidate(cx + bw, cy, cz)
                    add_candidate(cx, cy + bh, cz)
                    add_candidate(cx, cy, cz + bd)
                    
                    # Remove the used candidate? No, other boxes might use it?
                    # No, if a box is placed at (cx,cy,cz), that point is now occupied.
                    # But the *corner* of the new box creates new extreme points.
                    # The point (cx,cy,cz) itself is no longer an "empty" extreme point for a new box to start *at* if it overlaps?
                    # Actually, if a box is placed at P, P is inside the box.
                    # So we should remove P from candidates?
                    # Yes, because any new box starting at P would overlap the existing box (since P is the corner).
                    # Wait, if I place box A at (0,0,0) size (1,1,1).
                    # Candidate (0,0,0) is used.
                    # New candidates: (1,0,0), (0,1,0), (0,0,1).
                    # If I try to place box B at (0,0,0), it overlaps A.
                    # So we should remove (0,0,0) from candidates.
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _try_place_multi(boxes, container, seed=0):
    """
    Alternative placement strategy using different sorting criteria.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Different sorting: prefer smallest face area first?
        rots.sort(key=lambda r: (r[0]*r[1] + r[1]*r[2] + r[0]*r[2], r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    # Sort boxes by largest dimension first
    box_indices = list(range(n))
    box_max_dim = [max(w, h, d) for w, h, d in boxes]
    box_indices.sort(key=lambda i: -box_max_dim[i])
    
    placed = []
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    candidates = [(0, 0, 0)]
    
    def overlaps_any_local(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def add_candidate_local(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            found = False
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any_local(cx, cy, cz, bw, bh, bd):
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    add_candidate_local(cx + bw, cy, cz)
                    add_candidate_local(cx, cy + bh, cz)
                    add_candidate_local(cx, cy, cz + bd)
                    
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _evaluate_result(placement, container):
    """Evaluate the fraction of container volume filled."""
    W, H, D = container
    total_volume = W * H * D
    if total_volume == 0:
        return 0.0
    
    filled = 0
    for bin_id, x, y, z, w, h, d in placement:
        if bin_id == 0:
            # Check if placement is valid
            if x >= 0 and y >= 0 and z >= 0 and \
               x + w <= W and y + h <= H and z + d <= D:
                filled += w * h * d
    
    return filled / total_volume

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try multiple strategies and pick the best
    best_placement = None
    best_score = -1
    
    # Strategy 1: Volume descending
    if time.time() < deadline:
        placement1 = _try_place(boxes, container, sort_key=None)
        score1 = _evaluate_result(placement1, container)
        if score1 > best_score:
            best_score = score1
            best_placement = placement1
    
    # Strategy 2: Max dimension descending
    if time.time() < deadline:
        placement2 = _try_place_multi(boxes, container)
        score2 = _evaluate_result(placement2, container)
        if score2 > best_score:
            best_score = score2
            best_placement = placement2
    
    # Strategy 3: Surface area descending
    if time.time() < deadline:
        def sort_by_surface_area(i):
            w, h, d = boxes[i]
            return -2*(w*h + w*d + h*d)
        placement3 = _try_place(boxes, container, sort_key=sort_by_surface_area)
        score3 = _evaluate_result(placement3, container)
        if score3 > best_score:
            best_score = score3
            best_placement = placement3
    
    if best_placement is None:
        # Fallback: return all as left out
        best_placement = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placement
# EVOLVE-BLOCK-END
