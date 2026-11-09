"""3D Container Loading: multi-strategy greedy with rotations and local search."""
import time
import itertools
import random

# Deadline guard: kill before 150s wall-clock
_DEADLINE = 120.0  # safety margin

def _rotations(w, h, d):
    """Return all unique orientations of a box (up to 6)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _no_overlap(box_dims, pos, placed):
    """Check if box at pos with dims doesn't overlap any placed box."""
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _fits_in_container(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _extreme_points(placed, container):
    """Generate candidate positions from corners of placed boxes and container origin."""
    W, H, D = container
    candidates = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placed:
        # 6 candidate positions relative to this box
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
        candidates.append((px, py, pz + pd))
        candidates.append((px + pw, py + ph, pz))
        candidates.append((px + pw, py, pz + pd))
        candidates.append((px, py + ph, pz + pd))
        candidates.append((px + pw, py + ph, pz + pd))
    # Filter: must be within container
    valid = []
    seen = set()
    for (x, y, z) in candidates:
        if x < 0 or y < 0 or z < 0:
            continue
        if x >= W or y >= H or z >= D:
            continue
        if (x, y, z) not in seen:
            seen.add((x, y, z))
            valid.append((x, y, z))
    return valid

def _greedy_place(boxes, container, sort_key, time_limit):
    """
    Greedy placement: sort boxes by sort_key, try to place each at best extreme point.
    Returns list of (bin, x, y, z, w, h, d) per box.
    """
    W, H, D = container
    start_time = time.time()
    
    # Precompute rotations for each box
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort indices by the given key
    indexed = list(range(len(boxes)))
    indexed.sort(key=lambda i: sort_key(boxes[i], box_rots[i]), reverse=True)
    
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)
    
    for idx in indexed:
        if time.time() > time_limit:
            break
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = -1
        
        # Get candidate positions
        candidates = _extreme_points(placed, container)
        
        for rot in box_rots[idx]:
            rw, rh, rd = rot
            for (cx, cy, cz) in candidates:
                if not _fits_in_container((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                # Score: prefer lower z, then lower y, then lower x (compactness)
                # Also prefer tighter fit (smaller max dimension used)
                score = -(cz * 10000 + cy * 100 + cx) - (max(rw, rh, rd) * 0.1)
                if score > best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = rot
        
        if best_pos is not None:
            placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
            result[idx] = (0, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced boxes as left out
    for i in range(len(boxes)):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _volume_filled(placements, container):
    """Calculate fraction of container volume filled."""
    W, H, D = container
    total_vol = W * H * D
    if total_vol == 0:
        return 0.0
    filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return filled / total_vol

def _local_search(placements, boxes, container, time_limit):
    """
    Local search: try removing a box and reinserting it (possibly with different rotation).
    """
    W, H, D = container
    start_time = time.time()
    
    # Build placed list from placements
    placed = []
    for i, p in enumerate(placements):
        if p[0] == 0:
            placed.append((i, p[1], p[2], p[3], p[4], p[5], p[6]))
    
    # Precompute rotations
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try removing each placed box and reinserting
        for pi in range(len(placed)):
            if time.time() > time_limit:
                break
            idx, px, py, pz, pw, ph, pd = placed[pi]
            
            # Remove
            placed.pop(pi)
            
            # Try all rotations and extreme points
            best_pos = None
            best_rot = None
            best_score = -1
            
            candidates = _extreme_points(
                [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed],
                container
            )
            
            for rot in box_rots[idx]:
                rw, rh, rd = rot
                for (cx, cy, cz) in candidates:
                    if not _fits_in_container((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    if not _no_overlap((rw, rh, rd), (cx, cy, cz),
                                       [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed]):
                        continue
                    score = -(cz * 10000 + cy * 100 + cx)
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = rot
            
            if best_pos is not None:
                # Check if this improves the overall packing (simpler: just accept if it fits)
                # We accept any valid reposition for now
                placed.append((idx, best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
                improved = True
            else:
                # Reinsert at original position
                placed.append((idx, px, py, pz, pw, ph, pd))
    
    # Rebuild placements
    new_placements = [None] * len(boxes)
    for i in range(len(boxes)):
        w, h, d = boxes[i]
        new_placements[i] = (-1, 0, 0, 0, w, h, d)
    
    for (idx, x, y, z, w2, h2, d2) in placed:
        new_placements[idx] = (0, x, y, z, w2, h2, d2)
    
    return new_placements

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    time_limit = start_time + _DEADLINE - 5.0  # 5s safety margin
    
    # Define sorting strategies
    def sort_volume(box, rots):
        return box[0] * box[1] * box[2]
    
    def sort_max_dim(box, rots):
        return max(box)
    
    def sort_min_dim(box, rots):
        return min(box)
    
    def sort_aspect(box, rots):
        # Prefer boxes that are more "cubic" or fit well
        dims = sorted(box)
        return dims[0] * dims[2]  # product of min and max
    
    strategies = [
        ("volume_desc", sort_volume),
        ("max_dim_desc", sort_max_dim),
        ("min_dim_asc", lambda b, r: -min(b)),
        ("aspect", sort_aspect),
    ]
    
    best_placements = None
    best_score = -1
    
    for name, sort_key in strategies:
        if time.time() > time_limit:
            break
        
        placements = _greedy_place(boxes, container, sort_key, time_limit)
        score = _volume_filled(placements, container)
        
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # Local search on the best solution
    if best_placements is not None and time.time() < time_limit:
        improved = _local_search(best_placements, boxes, container, time_limit)
        improved_score = _volume_filled(improved, container)
        if improved_score > best_score:
            best_placements = improved
            best_score = improved_score
    
    return best_placements
