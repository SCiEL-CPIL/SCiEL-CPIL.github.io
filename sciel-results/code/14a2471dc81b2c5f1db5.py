"""Rewritten 3D container loading with extreme-point greedy + local search."""
import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _extreme_points(placed, container):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    # Filter to those within container
    xs = [x for x in xs if x < W]
    ys = [y for y in ys if y < H]
    zs = [z for z in zs if z < D]
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _greedy(boxes, container, order, deadline):
    """Greedy placement using extreme points for a given order. Returns (placements, volume)."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        best_pos = None
        best_rot = None
        # Try extreme points
        candidates = _extreme_points(placed, container)
        # Also try origin
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    best_pos = pos
                    best_rot = (rw, rh, rd)
                    break
            if best_pos is not None:
                break
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any unplaced
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def _greedy_with_heuristic(boxes, container, order, deadline, heuristic='extreme'):
    """Greedy placement with a specific heuristic for position selection."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        best_pos = None
        best_rot = None
        best_score = None
        
        if heuristic == 'extreme':
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        elif heuristic == 'faces':
            candidates = set()
            candidates.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd) in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            candidates = [(x, y, z) for (x, y, z) in candidates if x < W and y < H and z < D]
        else:
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    if heuristic == 'compact':
                        score = pos[0] + pos[1] + pos[2]
                    else:
                        score = pos[0] + pos[1] + pos[2]
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    W, H, D = container
    n = len(boxes)
    
    # Precompute volumes and sort indices
    vols = [w * h * d for w, h, d in boxes]
    
    # Initial greedy: largest first
    order = list(range(n))
    order.sort(key=lambda i: vols[i], reverse=True)
    
    best_placements, best_vol = _greedy(boxes, container, order, deadline)
    
    # Try different heuristics with the same order
    for heuristic in ['faces', 'compact']:
        if time.time() > deadline:
            break
        placements, vol = _greedy_with_heuristic(boxes, container, order, deadline, heuristic)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    # Local search: try multiple random orderings
    rng = random.Random(42)
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        # Create a perturbed order: mostly sorted, with some swaps
        order = list(range(n))
        order.sort(key=lambda i: vols[i], reverse=True)
        # Randomly swap a few elements
        for _ in range(min(5, n)):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                order[a], order[b] = order[b], order[a]
        
        placements, vol = _greedy(boxes, container, order, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    return best_placements
