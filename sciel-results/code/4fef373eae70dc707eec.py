"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(placements, boxes, container):
    """Compute the score: B + 0.1*(1 - last-bin occupied volume fraction)."""
    if not placements:
        return 0.0
    
    W, H, D = container
    container_vol = W * H * D
    
    bin_volumes = {}
    for (b, x, y, z, w, h, d) in placements:
        bin_volumes[b] = bin_volumes.get(b, 0) + w * h * d
    
    B = len(bin_volumes)
    if B == 0:
        return 0.0
    
    last_bin = max(bin_volumes.keys())
    last_vol = bin_volumes[last_bin]
    last_frac = last_vol / container_vol if container_vol > 0 else 0.0
    
    return B + 0.1 * (1.0 - last_frac)


def _find_position(bin_boxes, w, h, d, W, H, D):
    """Find a valid position in the bin for a box with dims (w,h,d).
    Returns (x, y, z) or None. Uses corner-candidate generation.
    Prefers positions that minimize z, then y, then x."""
    if not bin_boxes:
        return (0, 0, 0)
    
    # Generate candidate coordinates from existing box edges
    x_cands = {0}
    y_cands = {0}
    z_cands = {0}
    for (ox, oy, oz, ow, oh, od) in bin_boxes:
        x_cands.add(ox + ow)
        y_cands.add(oy + oh)
        z_cands.add(oz + od)
    
    x_cands = sorted([xc for xc in x_cands if xc + w <= W])
    y_cands = sorted([yc for yc in y_cands if yc + h <= H])
    z_cands = sorted([zc for zc in z_cands if zc + d <= D])
    
    if not x_cands or not y_cands or not z_cands:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    for xc in x_cands:
        for yc in y_cands:
            for zc in z_cands:
                # Fast overlap check
                ok = True
                for (ox, oy, oz, ow, oh, od) in bin_boxes:
                    if (xc < ox + ow and ox < xc + w and
                        yc < oy + oh and oy < yc + h and
                        zc < oz + od and oz < zc + d):
                        ok = False
                        break
                if ok:
                    # Prefer positions that are lower in z, then y, then x
                    pos_score = zc * 1000000 + yc * 1000 + xc
                    if pos_score < best_score:
                        best_score = pos_score
                        best_pos = (xc, yc, zc)
                    # Early exit if we found the absolute best possible position
                    if xc == 0 and yc == 0 and zc == 0:
                        return (0, 0, 0)
    
    return best_pos


def _pack_bfd(boxes, container, order_key, time_limit=None, start_time=None):
    """Best Fit Decreasing with a given ordering key.
    order_key: callable that takes (w,h,d) and returns sort key (ascending).
    We sort descending by this key."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 115 - (start_time - start_time)  # full budget
    
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by the key descending
    indexed = list(range(n))
    indexed.sort(key=lambda i: order_key(boxes[i]), reverse=True)
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    results = [None] * n
    
    for idx in indexed:
        if time.time() > start_time + time_limit:
            # Time's up, place remaining boxes at origin of new bins (safe fallback)
            for j in range(idx, n):
                if results[j] is None:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
                    results[j] = (b_idx, 0, 0, 0)
            break
        
        w, h, d = boxes[idx]
        placed = False
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = _find_position(bin_boxes, w, h, d, W, H, D)
            if pos is not None:
                x, y, z = pos
                bin_boxes.append((x, y, z, w, h, d))
                results[idx] = (b_idx, x, y, z)
                placed = True
                break
        
        if not placed:
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[idx] = (b_idx, 0, 0, 0)
    
    # Fill in any remaining unplaced boxes
    for j in range(n):
        if results[j] is None:
            b_idx = len(bins)
            bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
            results[j] = (b_idx, 0, 0, 0)
    
    placements = []
    for i in range(n):
        b_idx, x, y, z = results[i]
        w, h, d = boxes[i]
        placements.append((b_idx, x, y, z, w, h, d))
    
    return placements


def _pack_shelf(boxes, container):
    """Shelf heuristic: rows along x, shelves along y, layers along z."""
    W, H, D = container
    placements = []
    b = 0
    x = y = z = 0
    row_h = 0
    layer_d = 0
    for w, h, d in boxes:
        if x + w > W:
            x = 0
            y += row_h
            row_h = 0
        if y + h > H:
            y = 0
            z += layer_d
            layer_d = 0
            x = 0
            row_h = 0
        if z + d > D:
            b += 1
            x = y = z = 0
            row_h = 0
            layer_d = 0
        placements.append((b, x, y, z, w, h, d))
        x += w
        row_h = max(row_h, h)
        layer_d = max(layer_d, d)
    return placements


def _pack_with_time_budget(boxes, container, deadline):
    """Try multiple heuristics within the time budget, return best placement."""
    start_time = time.time()
    
    best_placements = None
    best_score = float('inf')
    
    def update(placements):
        nonlocal best_placements, best_score
        if placements is None:
            return
        score = _compute_score(placements, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # Strategy 1: Shelf (very fast)
    update(_pack_shelf(boxes, container))
    
    if time.time() > deadline - 5:
        return best_placements
    
    # Strategy 2: BFD sorted by volume descending
    update(_pack_bfd(boxes, container, lambda b: b[0]*b[1]*b[2]))
    
    if time.time() > deadline - 3:
        return best_placements
    
    # Strategy 3: BFD sorted by max dimension descending
    update(_pack_bfd(boxes, container, lambda b: max(b)))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 4: BFD sorted by volume, then max dim (tie-break)
    update(_pack_bfd(boxes, container, lambda b: b[0]*b[1]*b[2] + 0.001*max(b)))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 5: BFD sorted by surface area descending
    update(_pack_bfd(boxes, container, lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2])))
    
    if time.time() > deadline - 2:
        return best_placements
    
    # Strategy 6+: Randomized restarts with mixed ordering
    n = len(boxes)
    vol = [w * h * d for w, h, d in boxes]
    maxdim = [max(w, h, d) for w, h, d in boxes]
    surf = [2*(w*h + h*d + w*d) for w, h, d in boxes]
    
    attempts = 0
    while time.time() < deadline - 1:
        attempts += 1
        # Build a random-weighted ordering
        # Mix of volume, max-dim, and surface area with random perturbation
        w1 = random.random()
        w2 = random.random()
        w3 = random.random()
        total_w = w1 + w2 + w3
        if total_w == 0:
            total_w = 1
        
        weights = []
        for i in range(n):
            weight = (w1 * vol[i] + w2 * maxdim[i] * max(vol) / max(maxdim) + w3 * surf[i] * max(vol) / max(surf))
            weight += random.random() * 0.05 * max(vol)
            weights.append(weight)
        
        indexed = list(range(n))
        indexed.sort(key=lambda i: weights[i], reverse=True)
        
        W, H, D = container
        bins = []
        results = [None] * n
        
        for idx in indexed:
            if time.time() > deadline - 1:
                break
            w, h, d = boxes[idx]
            placed = False
            for b_idx, bin_boxes in enumerate(bins):
                pos = _find_position(bin_boxes, w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bin_boxes.append((x, y, z, w, h, d))
                    results[idx] = (b_idx, x, y, z)
                    placed = True
                    break
            if not placed:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                results[idx] = (b_idx, 0, 0, 0)
        
        # Fill in any remaining unplaced boxes
        for j in range(n):
            if results[j] is None:
                b_idx = len(bins)
                bins.append([(0, 0, 0, boxes[j][0], boxes[j][1], boxes[j][2])])
                results[j] = (b_idx, 0, 0, 0)
        
        placements = []
        for i in range(n):
            b_idx, x, y, z = results[i]
            w, h, d = boxes[i]
            placements.append((b_idx, x, y, z, w, h, d))
        
        update(placements)
        
        # Safety: stop if we've done many attempts
        if attempts > 100:
            break
    
    return best_placements


def pack(boxes, container):
    """
    Place every box into numbered bins.
    Returns one placement (bin, x, y, z, w, h, d) per box.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    if not boxes:
        return []
    
    placements = _pack_with_time_budget(boxes, container, deadline)
    
    return placements

# EVOLVE-BLOCK-END
