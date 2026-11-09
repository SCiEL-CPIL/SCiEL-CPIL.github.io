"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools
import math
from collections import defaultdict

def _rotations(w, h, d):
    """Return unique rotations of a box (w,h,d)."""
    return list(set(itertools.permutations([w, h, d])))

def _fits(p, placed, W, H, D):
    """Check if placement p=(x,y,z,w,h,d) fits in container and doesn't overlap placed boxes."""
    x, y, z, w, h, d = p
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # overlap check
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _best_fit_positions(w, h, d, placed, W, H, D):
    """Generate candidate positions for a box of size (w,h,d) given placed boxes.
    Uses the 'floor' heuristic: try positions at each existing box's top/side faces."""
    # Candidate x positions: 0 and right edge of each placed box
    xs = [0]
    ys = [0]
    zs = [0]
    for (px, py, pz, pw, ph, pd) in placed:
        xs.append(px + pw)
        ys.append(py + ph)
        zs.append(pz + pd)
    
    # Deduplicate and filter
    xs = sorted(set(x for x in xs if x + w <= W))
    ys = sorted(set(y for y in ys if y + h <= H))
    zs = sorted(set(z for z in zs if z + d <= D))
    
    # Also add positions just after each box in each axis
    # We already have those. Now generate all combos and check fit.
    # To limit candidates, we can be smarter: only consider positions where
    # the box touches at least one placed box or a wall.
    
    candidates = []
    for x in xs:
        for y in ys:
            for z in zs:
                p = (x, y, z, w, h, d)
                if _fits(p, placed, W, H, D):
                    # Score: prefer lower z, then lower y, then lower x (pack from bottom-front-left)
                    score = (z, y, x)
                    candidates.append((score, p))
    
    # Sort by score and return unique positions
    candidates.sort(key=lambda t: t[0])
    seen = set()
    result = []
    for score, p in candidates:
        key = (p[0], p[1], p[2])
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result

def _greedy_pack(boxes, container, deadline):
    """Greedy First-Fit Decreasing with rotations, using floor heuristic for positions."""
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda t: -t[4])
    
    placed = []  # list of (x,y,z,w,h,d)
    placements = [None] * len(boxes)
    
    start_time = time.time()
    
    for i, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            break
        for rw, rh, rd in _rotations(w, h, d):
            positions = _best_fit_positions(rw, rh, rd, placed, W, H, D)
            if positions:
                # Take the first (best) position
                x, y, z, pw, ph, pd = positions[0]
                placed.append((x, y, z, pw, ph, pd))
                placements[i] = (0, x, y, z, pw, ph, pd)
                break
        if placements[i] is None:
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any remaining as left out
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements

def _local_search(placements, boxes, container, deadline):
    """Simple local search: try to re-insert left-out boxes by removing some placed boxes."""
    W, H, D = container
    container_vol = W * H * D
    
    # Parse current placements
    placed_indices = []  # (box_idx, x, y, z, w, h, d)
    left_out_indices = []
    for i, p in enumerate(placements):
        if p[0] == 0:
            placed_indices.append((i, p[1], p[2], p[3], p[4], p[5], p[6]))
        else:
            left_out_indices.append(i)
    
    if not left_out_indices:
        return placements
    
    start_time = time.time()
    best_placements = list(placements)
    best_volume = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
    
    # Try to insert each left-out box by potentially removing placed boxes
    for lo_idx in left_out_indices:
        if time.time() > deadline:
            break
        w, h, d = boxes[lo_idx]
        for rw, rh, rd in _rotations(w, h, d):
            # Try to find a position for this box, possibly removing overlapping placed boxes
            # Simple approach: try all positions, and if it overlaps, check if removing those boxes is worth it
            positions = _best_fit_positions(rw, rh, rd, [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in placed_indices], W, H, D)
            if positions:
                x, y, z, pw, ph, pd = positions[0]
                # Check if this position is actually free (it should be, from _best_fit_positions)
                # Update placements
                new_placements = list(best_placements)
                new_placements[lo_idx] = (0, x, y, z, pw, ph, pd)
                new_volume = best_volume + pw*ph*pd
                if new_volume > best_volume:
                    best_volume = new_volume
                    best_placements = new_placements
                    # Update placed_indices
                    placed_indices = []
                    for i, p in enumerate(best_placements):
                        if p[0] == 0:
                            placed_indices.append((i, p[1], p[2], p[3], p[4], p[5], p[6]))
    
    return best_placements

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline: finish by 140s (10s safety margin)
    deadline = time.time() + 140
    
    # Phase 1: Greedy packing
    placements = _greedy_pack(boxes, container, deadline)
    
    if time.time() > deadline - 5:
        return placements
    
    # Phase 2: Local search
    placements = _local_search(placements, boxes, container, deadline)
    
    return placements
# EVOLVE-BLOCK-END
