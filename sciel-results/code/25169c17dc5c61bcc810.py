"""3D Bin Packing: place boxes unrotated into numbered bins, minimize bins + last-bin fill penalty."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import defaultdict

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Index boxes by original index to preserve return order
    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    def volume(w, h, d):
        return w * h * d

    # Try multiple sorting strategies, keep best
    strategies = [
        lambda items: sorted(items, key=lambda x: -volume(x[1], x[2], x[3])),          # by volume desc
        lambda items: sorted(items, key=lambda x: -(x[1]*x[2]*x[3])),                    # same
        lambda items: sorted(items, key=lambda x: -(max(x[1],x[2],x[3]))),               # by max dim
        lambda items: sorted(items, key=lambda x: -(x[1]+x[2]+x[3])),                    # by sum dims
        lambda items: sorted(items, key=lambda x: -(x[1]*x[2], x[3])),                   # by base area
        lambda items: sorted(items, key=lambda x: -(x[2]*x[3], x[1])),                   # by side area
        lambda items: sorted(items, key=lambda x: -(x[1]*x[3], x[2])),                   # by other side
        lambda items: sorted(items, key=lambda x: (-volume(x[1],x[2],x[3]), -max(x[1],x[2],x[3]))),
    ]

    def can_place(bin_items, w, h, d, x, y, z):
        """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_items:
            # Overlap check: boxes overlap if they intersect in all 3 axes
            if (x < bx + bw and x + w > bx) and \
               (y < by + bh and y + h > by) and \
               (z < bz + bd and z + d > bz):
                return False
        return True

    def find_spot(bin_items, w, h, d):
        """Find a valid (x,y,z) for box in this bin. Try candidate positions."""
        # Generate candidate x, y, z positions from existing box edges and 0
        xs = {0}
        ys = {0}
        zs = {0}
        for (bx, by, bz, bw, bh, bd) in bin_items:
            if bx + bw <= W:
                xs.add(bx + bw)
            if by + bh <= H:
                ys.add(by + bh)
            if bz + bd <= D:
                zs.add(bz + bd)
        # Also try positions just after existing boxes
        for (bx, by, bz, bw, bh, bd) in bin_items:
            if bx + bw + w <= W:
                xs.add(bx + bw)
            if by + bh + h <= H:
                ys.add(by + bh)
            if bz + bd + d <= D:
                zs.add(bz + bd)

        # Sort candidates to prefer lower positions (better packing)
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)

        best_pos = None
        best_score = float('inf')
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_items, w, h, d, x, y, z):
                        # Prefer positions that minimize "wasted space" - use sum of coords as heuristic
                        score = x + y * (W+1) + z * (W+1)*(H+1)
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        return best_pos

    def try_order(order):
        """Try to pack boxes in given order. Returns placements dict or None if failed."""
        bins = []  # list of lists of (x,y,z,w,h,d)
        placements = {}  # index -> (bin, x, y, z, w, h, d)
        
        for idx, w, h, d in order:
            placed = False
            # Try existing bins first (best fit: minimize wasted space)
            best_bin = -1
            best_pos = None
            best_waste = float('inf')
            
            for bi in range(len(bins)):
                pos = find_spot(bins[bi], w, h, d)
                if pos is not None:
                    # Calculate waste: prefer bins where box fits more tightly
                    # Use the position's "distance from origin" as a proxy for compactness
                    x, y, z = pos
                    waste = x + y + z
                    if waste < best_waste:
                        best_waste = waste
                        best_bin = bi
                        best_pos = pos
            
            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                placements[idx] = (best_bin, x, y, z, w, h, d)
            else:
                # New bin
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[idx] = (bi, 0, 0, 0, w, h, d)
        
        return placements

    def score_placements(placements):
        """Compute score: B + 0.1*(1 - last_bin_fill_fraction)"""
        if not placements:
            return 0.0
        bin_ids = set(p[0] for p in placements.values())
        B = len(bin_ids)
        last_bin = max(bin_ids)
        # Compute volume in last bin
        last_vol = 0
        for p in placements.values():
            if p[0] == last_bin:
                last_vol += p[4] * p[5] * p[6]
        container_vol = W * H * D
        if container_vol == 0:
            fill_frac = 1.0
        else:
            fill_frac = last_vol / container_vol
        return B + 0.1 * (1 - fill_frac)

    def to_output(placements, n):
        """Convert placements dict to list in original box order."""
        result = []
        for i in range(n):
            result.append(placements[i])
        return result

    best_placements = None
    best_score_val = float('inf')

    for si, strat in enumerate(strategies):
        if time.time() > deadline:
            break
        try:
            order = strat(indexed)
            placements = try_order(order)
            if placements is None or len(placements) != n:
                continue
            s = score_placements(placements)
            if s < best_score_val:
                best_score_val = s
                best_placements = placements
        except Exception:
            continue

    # If we still have time, try some random permutations for improvement
    if best_placements is not None and time.time() < deadline - 5.0:
        # Try a few random shuffles of the best strategy's order
        base_order = strategies[0](indexed)
        for trial in range(50):
            if time.time() > deadline:
                break
            shuffled = base_order[:]
            random.shuffle(shuffled)
            try:
                placements = try_order(shuffled)
                if placements is None or len(placements) != n:
                    continue
                s = score_placements(placements)
                if s < best_score_val:
                    best_score_val = s
                    best_placements = placements
            except Exception:
                continue

    # Fallback: simple shelf if everything failed
    if best_placements is None:
        best_placements = {}
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements[i] = (b, x, y, z, w, h, d)
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return to_output(best_placements, n)

# EVOLVE-BLOCK-END
