"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    seen = set()
    out = []
    for p in permutations((w, h, d)):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _fits(x, y, z, bw, bh, bd, placed, W, H, D):
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True

def _candidate_positions(placed, W, H, D):
    """Generate candidate (x,y,z) positions: origin + face-adjacent points."""
    pts = {(0, 0, 0)}
    for (px, py, pz, pw, ph, pd) in placed:
        if px + pw < W:
            pts.add((px + pw, py, pz))
        if py + ph < H:
            pts.add((px, py + ph, pz))
        if pz + pd < D:
            pts.add((px, py, pz + pd))
        # also try combinations of two offsets
        if px + pw < W and py + ph < H:
            pts.add((px + pw, py + ph, pz))
        if px + pw < W and pz + pd < D:
            pts.add((px + pw, py, pz + pd))
        if py + ph < H and pz + pd < D:
            pts.add((px, py + ph, pz + pd))
    return list(pts)

def _best_position(bw, bh, bd, placed, W, H, D):
    """Find the best (x,y,z) for a box, preferring lower z, then y, then x."""
    candidates = _candidate_positions(placed, W, H, D)
    best = None
    for (x, y, z) in candidates:
        if _fits(x, y, z, bw, bh, bd, placed, W, H, D):
            key = (z, y, x)
            if best is None or key < best[0]:
                best = (key, (x, y, z))
    if best is not None:
        return best[1]
    return None

def _greedy_pack(order, boxes, container, deadline, rotations_cache):
    """Greedy pack boxes in given order, trying all rotations. Returns (placements, volume)."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    volume = 0
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = rotations_cache[idx]
        best_pos = None
        best_rot = None
        # Try rotations, prefer the one that fits at the "best" position
        for (rw, rh, rd) in rots:
            pos = _best_position(rw, rh, rd, placed, W, H, D)
            if pos is not None:
                key = (pos[2], pos[1], pos[0])
                if best_pos is None or key < best_pos[0]:
                    best_pos = (key, pos)
                    best_rot = (rw, rh, rd)
        if best_pos is not None:
            x, y, z = best_pos[1]
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            volume += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    # Fill any remaining None (shouldn't happen if we broke on deadline)
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    return placements, volume

def _make_order(boxes, strategy, rng):
    n = len(boxes)
    if strategy == 0:  # largest volume first
        return sorted(range(n), key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    elif strategy == 1:  # largest max-dim first
        return sorted(range(n), key=lambda i: -max(boxes[i]))
    elif strategy == 2:  # smallest volume first
        return sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    elif strategy == 3:  # most "flat" first (max/min ratio)
        return sorted(range(n), key=lambda i: -(max(boxes[i]) / min(boxes[i])))
    elif strategy == 4:  # random
        return list(range(n))
    elif strategy == 5:  # compactness (min dimension) descending
        return sorted(range(n), key=lambda i: -min(boxes[i]))
    else:
        return list(range(n))

def _try_improve(placements, boxes, container, deadline):
    """Try to fit left-out boxes by reordering: move a placed box out and try to fit a left-out one."""
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    current_vol = sum(w*h*d for (b,x,y,z,w,h,d) in placements if b == 0)
    
    # Build list of placed indices and left-out indices
    placed_idx = [i for i in range(n) if placements[i][0] == 0]
    left_idx = [i for i in range(n) if placements[i][0] == -1]
    
    if not left_idx:
        return placements, current_vol
    
    # For each left-out box, try to find a position by temporarily removing placed boxes
    # This is expensive; do a limited pass
    improved = False
    for li in left_idx[:10]:  # limit to first 10 left-out
        if time.time() > deadline:
            break
        w, h, d = boxes[li]
        rots = _rotations(w, h, d)
        # Try to fit by removing one placed box at a time
        for pi in placed_idx[:20]:
            if time.time() > deadline:
                break
            pw, ph, pd = placements[pi][4], placements[pi][5], placements[pi][6]
            # Temporarily remove box pi
            placed_list = [placements[j] for j in range(n) if placements[j][0] == 0 and j != pi]
            placed_tuples = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in placed_list]
            for (rw, rh, rd) in rots:
                pos = _best_position(rw, rh, rd, placed_tuples, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    # Now try to re-place box pi
                    pi_rots = _rotations(pw, ph, pd)
                    placed_with_new = placed_tuples + [(x, y, z, rw, rh, rd)]
                    pi_pos = None
                    for (prw, prh, prd) in pi_rots:
                        ppos = _best_position(prw, prh, prd, placed_with_new, W, H, D)
                        if ppos is not None:
                            pi_pos = (ppos, (prw, prh, prd))
                            break
                    if pi_pos is not None:
                        px, py, pz = pi_pos[0]
                        prw, prh, prd = pi_pos[1]
                        # Update placements
                        placements[li] = (0, x, y, z, rw, rh, rd)
                        placements[pi] = (0, px, py, pz, prw, prh, prd)
                        current_vol += rw*rh*rd
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return placements, current_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    # Precompute rotations
    rotations_cache = [_rotations(w, h, d) for (w, h, d) in boxes]
    
    best_placements = None
    best_volume = -1
    
    # Strategy list: (strategy_id, description)
    strategies = [0, 1, 2, 3, 4, 5]
    
    # First pass: deterministic strategies
    for strat in strategies:
        if time.time() > deadline:
            break
        order = _make_order(boxes, strat, None)
        placements, vol = _greedy_pack(order, boxes, container, deadline, rotations_cache)
        if vol > best_volume:
            best_volume = vol
            best_placements = [list(p) for p in placements]
    
    # Second pass: random restarts
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline:
        order = list(range(n))
        rng.shuffle(order)
        # Bias: larger boxes earlier
        order.sort(key=lambda i: -rng.random() * boxes[i][0]*boxes[i][1]*boxes[i][2])
        placements, vol = _greedy_pack(order, boxes, container, deadline, rotations_cache)
        if vol > best_volume:
            best_volume = vol
            best_placements = [list(p) for p in placements]
        restart_count += 1
        if restart_count > 500:  # safety cap
            break
    
    # Third pass: try to improve the best solution
    if best_placements is not None and time.time() < deadline:
        best_placements, best_volume = _try_improve(
            [tuple(p) for p in best_placements], boxes, container, deadline
        )
    
    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]
    
    return [tuple(p) for p in best_placements]

# EVOLVE-BLOCK-END
