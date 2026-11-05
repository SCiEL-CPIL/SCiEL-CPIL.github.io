"""Structural change: Best-Fit-Decreasing 3D bin packing with corner-based placement."""
import time
import random

DEADLINE = time.time() + 145.0  # safety margin under 150s wall-clock


def _corners_for_bin(bin_state, W, H, D):
    """Generate candidate corners from existing boxes in the bin."""
    corners = set()
    # Origin
    corners.add((0, 0, 0))
    # For each box, add the corner opposite to it (x+w, y, z), (x, y+h, z), (x, y, z+d), etc.
    for (px, py, pz, pw, ph, pd) in bin_state['boxes']:
        corners.add((px + pw, py, pz))
        corners.add((px, py + ph, pz))
        corners.add((px, py, pz + pd))
        # Also add combinations
        corners.add((px + pw, py + ph, pz))
        corners.add((px + pw, py, pz + pd))
        corners.add((px, py + ph, pz + pd))
        corners.add((px + pw, py + ph, pz + pd))
    return list(corners)


def _fits(bin_state, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box of size (w,h,d) fits at position (cx,cy,cz) in the bin."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in bin_state['boxes']:
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True


def _best_fit_position(bin_state, w, h, d, W, H, D):
    """Find the best (lowest z, then y, then x) position for a box in this bin."""
    corners = _corners_for_bin(bin_state, W, H, D)
    # Sort corners by (z, y, x) to prefer lower positions
    corners.sort(key=lambda c: (c[2], c[1], c[0]))
    best_pos = None
    best_key = None
    for (cx, cy, cz) in corners:
        if _fits(bin_state, cx, cy, cz, w, h, d, W, H, D):
            key = (cz, cy, cx)
            if best_key is None or key < best_key:
                best_key = key
                best_pos = (cx, cy, cz)
            # We could break early if we find (0,0,0) but let's keep searching for now
            # Actually, since sorted by (z,y,x), the first valid one is the best
            break
    return best_pos


def _pack_bfd(boxes, container, order):
    """Pack boxes using Best-Fit-Decreasing with the given order."""
    W, H, D = container
    n = len(boxes)
    # bins: list of dicts with 'boxes' (list of placements) and 'volume'
    bins = []
    
    for idx in order:
        w, h, d = boxes[idx]
        vol = w * h * d
        
        best_bin = -1
        best_pos = None
        best_waste = float('inf')
        
        for bi in range(len(bins)):
            pos = _best_fit_position(bins[bi], w, h, d, W, H, D)
            if pos is not None:
                # Compute "waste" metric: prefer bins that leave less space
                # Use the z-coordinate of placement as a proxy for how full the bin is
                # Lower z means better fit
                waste = pos[2] * 10000 + pos[1] * 100 + pos[0]
                if waste < best_waste:
                    best_waste = waste
                    best_bin = bi
                    best_pos = pos
        
        if best_bin == -1:
            # Start a new bin
            bins.append({'boxes': [], 'volume': 0})
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        cx, cy, cz = best_pos
        bins[best_bin]['boxes'].append((cx, cy, cz, w, h, d))
        bins[best_bin]['volume'] += vol
    
    # Build placements
    placements = []
    for bi, b in enumerate(bins):
        for (cx, cy, cz, w, h, d) in b['boxes']:
            placements.append((bi, cx, cy, cz, w, h, d))
    return placements


def _score(boxes, container, placements):
    """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    W, H, D = container
    container_vol = W * H * D
    
    # Find occupied bins and last bin
    bin_vols = {}
    for (b, x, y, z, w, h, d) in placements:
        if b not in bin_vols:
            bin_vols[b] = 0
        bin_vols[b] += w * h * d
    
    if not bin_vols:
        return 0
    
    occupied_bins = len(bin_vols)
    last_bin = max(bin_vols.keys())
    last_bin_vol = bin_vols[last_bin]
    
    # Score for this instance
    score = occupied_bins + 0.1 * (1 - last_bin_vol / container_vol)
    return score


def pack(boxes, container):
    """Pack boxes into bins, returning best placement found within time budget."""
    W, H, D = container
    n = len(boxes)
    
    # Initial deadline
    deadline = time.time() + 145.0
    
    # Generate several orderings to try
    # 1. Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # 2. Max dimension descending
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    
    # 3. Surface area descending
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    # 4. Random shuffles
    rng = random.Random(42)
    random_orders = []
    for _ in range(5):
        order = list(range(n))
        rng.shuffle(order)
        random_orders.append(order)
    
    # Candidate orderings
    orderings = [vol_order, maxdim_order, sa_order] + random_orders
    
    best_placements = None
    best_score = float('inf')
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        placements = _pack_bfd(boxes, container, order)
        s = _score(boxes, container, placements)
        if s < best_score:
            best_score = s
            best_placements = placements
    
    # If we have time, try more random orderings
    if best_placements is None:
        # Fallback: simple shelf
        best_placements = _shelf_pack(boxes, container)
    else:
        # Try additional random orderings with time remaining
        while time.time() < deadline - 2:
            order = list(range(n))
            rng.shuffle(order)
            placements = _pack_bfd(boxes, container, order)
            s = _score(boxes, container, placements)
            if s < best_score:
                best_score = s
                best_placements = placements
    
    return best_placements


def _shelf_pack(boxes, container):
    """Simple shelf packing as fallback."""
    W, H, D = container
    placements = []
    b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
    for w, h, d in boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
        placements.append((b, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return placements
