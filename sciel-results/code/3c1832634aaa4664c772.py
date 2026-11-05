import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s kill

    # Pre-compute volumes
    volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def fits(x, y, z, w, h, d, bin_boxes):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ex, ey, ez, ew, eh, ed) in bin_boxes:
            if overlaps(x, y, z, w, h, d, ex, ey, ez, ew, eh, ed):
                return False
        return True

    def get_candidates(w, h, d, bin_boxes):
        """Generate candidate positions for a box in a bin."""
        candidates = set()
        candidates.add((0, 0, 0))
        for (ex, ey, ez, ew, eh, ed) in bin_boxes:
            candidates.add((ex + ew, ey, ez))
            candidates.add((ex, ey + eh, ez))
            candidates.add((ex, ey, ez + ed))
            # Also try combinations of offsets
            candidates.add((ex + ew, ey + eh, ez))
            candidates.add((ex + ew, ey, ez + ed))
            candidates.add((ex, ey + eh, ez + ed))
            candidates.add((ex + ew, ey + eh, ez + ed))
        
        # Filter valid candidates
        valid = []
        for (cx, cy, cz) in candidates:
            if cx + w <= W and cy + h <= H and cz + d <= D:
                if fits(cx, cy, cz, w, h, d, bin_boxes):
                    valid.append((cx, cy, cz))
        return valid

    def bin_score(bin_boxes, w, h, d, pos):
        """Score a placement: prefer positions that minimize wasted space."""
        cx, cy, cz = pos
        # Primary: minimize distance from origin (corner packing)
        # Secondary: prefer positions that leave larger contiguous spaces
        score = cx + cy + cz
        return score

    def pack_single_pass(order):
        """Try to pack boxes in the given order. Returns placements or None if timeout."""
        bins = []  # List of lists of (x, y, z, w, h, d)
        placements = [None] * n

        for idx in order:
            if time.time() > DEADLINE:
                return None

            w, h, d = boxes[idx]
            if w > W or h > H or d > D:
                return None  # Box doesn't fit in any bin

            best_bin = -1
            best_pos = None
            best_s = float('inf')

            for b_idx, bin_boxes in enumerate(bins):
                candidates = get_candidates(w, h, d, bin_boxes)
                for pos in candidates:
                    s = bin_score(bin_boxes, w, h, d, pos)
                    if s < best_s:
                        best_s = s
                        best_bin = b_idx
                        best_pos = pos

            if best_bin != -1:
                bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
                placements[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            else:
                bins.append([(0, 0, 0, w, h, d)])
                placements[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

        return placements

    def compute_score(placements):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
        if placements is None:
            return float('inf')

        # Count bins and compute last-bin volume
        bin_vols = {}
        for (b, x, y, z, w, h, d) in placements:
            if b not in bin_vols:
                bin_vols[b] = 0
            bin_vols[b] += w * h * d

        if not bin_vols:
            return 0.0

        B = len(bin_vols)
        last_bin = max(bin_vols.keys())
        last_vol = bin_vols[last_bin]
        last_frac = last_vol / container_vol if container_vol > 0 else 1.0

        return B + 0.1 * (1.0 - last_frac)

    # Strategy 1: Sort by volume descending
    order_vol = sorted(range(n), key=lambda i: -volumes[i])
    # Strategy 2: Sort by max dimension descending, then volume descending
    order_maxdim = sorted(range(n), key=lambda i: (-(max(boxes[i])), -volumes[i]))
    # Strategy 3: Sort by volume ascending (small first)
    order_vol_asc = sorted(range(n), key=lambda i: volumes[i])
    # Strategy 4: Random order
    order_rand = list(range(n))
    random.shuffle(order_rand)
    # Strategy 5: Sort by surface area descending
    order_surf = sorted(range(n), key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    # Strategy 6: Sort by min dimension descending
    order_mindim = sorted(range(n), key=lambda i: (-min(boxes[i]), -volumes[i]))

    strategies = [order_vol, order_maxdim, order_surf, order_mindim, order_vol_asc, order_rand]

    best_placements = None
    best_score = float('inf')

    for order in strategies:
        if time.time() > DEADLINE:
            break

        placements = pack_single_pass(order)
        if placements is None:
            continue

        s = compute_score(placements)
        if s < best_score:
            best_score = s
            best_placements = placements

    # If we still have time, try a few more random orders
    while time.time() < DEADLINE - 10.0:
        order = list(range(n))
        random.shuffle(order)
        placements = pack_single_pass(order)
        if placements is None:
            break
        s = compute_score(placements)
        if s < best_score:
            best_score = s
            best_placements = placements

    if best_placements is None:
        # Fallback: one box per bin
        best_placements = [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    return best_placements
