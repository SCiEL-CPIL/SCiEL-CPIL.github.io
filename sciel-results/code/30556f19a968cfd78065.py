"""Rewritten 3D container loading with stronger heuristic."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _orientations(w, h, d):
    """Return unique orientations (w,h,d) as tuples."""
    dims = (w, h, d)
    seen = set()
    result = []
    for perm in itertools.permutations(dims):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _can_place(px, py, pz, bw, bh, bd, placed, W, H, D):
    """Check if box at (px,py,pz) with dims (bw,bh,bd) doesn't overlap any placed box."""
    for (qx, qy, qz, qw, qh, qd) in placed:
        if (px < qx + qw and px + bw > qx and
            py < qy + qh and py + bh > qy and
            pz < qz + qd and pz + bd > qz):
            return False
    # Check bounds
    if px + bw > W or py + bh > H or pz + bd > D:
        return False
    return True


def _find_position(box_dims, placed, W, H, D):
    """Find the best position for a box with given dimensions.
    Returns (x, y, z, bw, bh, bd) or None."""
    best_pos = None
    best_score = None
    
    bw, bh, bd = box_dims
    
    # Generate candidate positions from all box-face corners plus axis-aligned boundaries
    candidates = set()
    
    # Origin
    candidates.add((0, 0, 0))
    
    # For each placed box, generate all 8 corners of its bounding box
    for (qx, qy, qz, qw, qh, qd) in placed:
        for dx in (0, qw):
            for dy in (0, qh):
                for dz in (0, qd):
                    candidates.add((qx + dx, qy + dy, qz + dz))
    
    # Also add positions aligned with container boundaries
    for (qx, qy, qz, qw, qh, qd) in placed:
        # Align to container edges
        candidates.add((0, qy, qz))
        candidates.add((qx, 0, qz))
        candidates.add((qx, qy, 0))
        candidates.add((W - bw, qy, qz)) if W - bw >= 0 else None
        candidates.add((qx, H - bh, qz)) if H - bh >= 0 else None
        candidates.add((qx, qy, D - bd)) if D - bd >= 0 else None
    
    # Filter valid candidates and find best
    for (px, py, pz) in candidates:
        if px < 0 or py < 0 or pz < 0:
            continue
        if px + bw > W or py + bh > H or pz + bd > D:
            continue
        if not _can_place(px, py, pz, bw, bh, bd, placed, W, H, D):
            continue
        # Score: prefer lower z (bottom-up), then lower y, then lower x
        score = (pz, py, px)
        if best_score is None or score < best_score:
            best_score = score
            best_pos = (px, py, pz)
    
    if best_pos is None:
        return None
    
    return (best_pos[0], best_pos[1], best_pos[2], bw, bh, bd)


def _greedy_pack(boxes, container, order, deadline):
    """Greedy pack boxes in given order. Returns placements list."""
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = [None] * len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        orients = _orientations(w, h, d)
        
        # Try orientations, pick the one that fits in a valid position
        best_result = None
        best_orient_score = None
        
        for orient in orients:
            bw, bh, bd = orient
            result = _find_position(orient, placed, W, H, D)
            if result is not None:
                px, py, pz, rbw, rbh, rbd = result
                # Score: prefer positions closer to origin, lower z
                score = (pz, py, px)
                if best_orient_score is None or score < best_orient_score:
                    best_orient_score = score
                    best_result = (px, py, pz, rbw, rbh, rbd)
        
        if best_result is not None:
            px, py, pz, bw, bh, bd = best_result
            placed.append((px, py, pz, bw, bh, bd))
            placements[idx] = (0, px, py, pz, bw, bh, bd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill any remaining None entries
    for i in range(len(placements)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements


def _score_placements(placements, boxes, container):
    """Compute the fraction of container volume filled."""
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return 0.0
    filled = 0
    for i, p in enumerate(placements):
        if p[0] == 0:
            bw, bh, bd = p[4], p[5], p[6]
            filled += bw * bh * bd
    return filled / container_vol


def load(boxes, container):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin for 150s limit
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if container_vol == 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Pre-compute orientations for each box
    all_orients = [_orientations(w, h, d) for w, h, d in boxes]
    
    # Strategy: try multiple orderings, keep best
    strategies = []
    
    # Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    strategies.append(vol_order)
    
    # Max dimension descending
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(maxdim_order)
    
    # Surface area descending
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append(sa_order)
    
    # Min dimension descending (helps with tight fits)
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    strategies.append(mindim_order)
    
    # Volume-to-max-dim ratio (prefer boxes that are "chunky")
    ratio_order = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(boxes[i]), reverse=True)
    strategies.append(ratio_order)
    
    # Try each strategy
    best_placements = None
    best_score = -1
    
    for order in strategies:
        if time.time() > DEADLINE - 5.0:
            break
        placements = _greedy_pack(boxes, container, order, DEADLINE)
        score = _score_placements(placements, boxes, container)
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # If time remains, try a few random orderings
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5.0:
        rng = random.Random(42)
        num_random = min(15, int(remaining_time / 2.0))
        for _ in range(num_random):
            if time.time() > DEADLINE - 2.0:
                break
            rand_order = list(range(n))
            rng.shuffle(rand_order)
            placements = _greedy_pack(boxes, container, rand_order, DEADLINE)
            score = _score_placements(placements, boxes, container)
            if score > best_score:
                best_score = score
                best_placements = placements
    
    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements

# EVOLVE-BLOCK-END
