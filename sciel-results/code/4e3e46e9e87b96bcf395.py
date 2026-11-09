"""3D Container Loading - improved greedy with extreme points and rotations."""
import time
import itertools
import random

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    """Generate candidate positions from placed boxes."""
    W, H, D = container
    points = set()
    # Origin
    points.add((0, 0, 0))
    # For each placed box, generate 8 corner-based candidate positions
    for (px, py, pz, pw, ph, pd) in placed:
        candidates = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px + pw, py, pz + pd),
            (px, py + ph, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for c in candidates:
            if 0 <= c[0] < W and 0 <= c[1] < H and 0 <= c[2] < D:
                points.add(c)
    return list(points)

def _score_position(pos, box_dims, placed, container):
    """Score a position: prefer positions closer to origin (compactness)."""
    x, y, z = pos
    # Prefer lower z, then lower y, then lower x for compactness
    W, H, D = container
    return z * (W * H) + y * W + x

def _greedy_load(boxes, container, deadline, seed=42):
    """Greedy placement with extreme points and rotation selection."""
    rng = random.Random(seed)
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)
    
    # Sort boxes by volume descending for better packing
    box_indices = list(range(len(boxes)))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in box_indices]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # Precompute rotations for each box
    box_rots = [_rotations(*boxes[i]) for i in range(len(boxes))]
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Get candidate positions
        candidates = _extreme_points(placed, container)
        
        # Limit candidates for speed
        if len(candidates) > 500:
            # Sort by proximity to origin and take top 500
            candidates.sort(key=lambda p: p[2] * (W * H) + p[1] * W + p[0])
            candidates = candidates[:500]
        
        for rot in box_rots[idx]:
            rw, rh, rd = rot
            for pos in candidates:
                if not _fits(rot, pos, container):
                    continue
                if not _no_overlap(rot, pos, placed):
                    continue
                sc = _score_position(pos, rot, placed, container)
                if sc < best_score:
                    best_score = sc
                    best_pos = pos
                    best_rot = rot
        
        if best_pos is not None:
            x, y, z = best_pos
            placed.append((x, y, z, best_rot[0], best_rot[1], best_rot[2]))
            result[idx] = (0, x, y, z, best_rot[0], best_rot[1], best_rot[2])
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced boxes
    for i in range(len(boxes)):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _simple_greedy(boxes, container):
    """Simple shelf-based greedy as fallback."""
    W, H, D = container
    placements = []
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
    for w, h, d in boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + d > D:
            placements.append((-1, 0, 0, 0, w, h, d))
            continue
        placements.append((0, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return placements

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    # Try multiple strategies and pick the best
    best_result = None
    best_volume = -1
    
    # Strategy 1: Simple greedy (fast baseline)
    result1 = _simple_greedy(boxes, container)
    vol1 = sum(w*h*d for (bin_, x, y, z, w, h, d) in result1 if bin_ == 0)
    if vol1 > best_volume:
        best_volume = vol1
        best_result = result1
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Extreme point greedy with rotations
    result2 = _greedy_load(boxes, container, deadline, seed=42)
    vol2 = sum(w*h*d for (bin_, x, y, z, w, h, d) in result2 if bin_ == 0)
    if vol2 > best_volume:
        best_volume = vol2
        best_result = result2
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 3: Extreme point greedy with different seed
    result3 = _greedy_load(boxes, container, deadline, seed=123)
    vol3 = sum(w*h*d for (bin_, x, y, z, w, h, d) in result3 if bin_ == 0)
    if vol3 > best_volume:
        best_volume = vol3
        best_result = result3
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 4: Sort by different criteria
    # Try sorting by max dimension
    sorted_boxes = sorted(boxes, key=lambda b: max(b), reverse=True)
    # We need to track original indices, so let's do a custom approach
    # For now, just use the best result we have
    
    return best_result
