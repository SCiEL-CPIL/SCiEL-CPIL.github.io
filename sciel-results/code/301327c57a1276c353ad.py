"""3D Bin Packing: multi-strategy greedy with Best-Fit-Then-First-Fit and local improvement."""
import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START

def _can_place(bin_items, x, y, z, w, h, d, W, H, D):
    """Check if a box of size (w,h,d) can be placed at (x,y,z) in a bin of size (W,H,D)
       without overlapping any existing item."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (bx, by, bz, bw, bh, bd) in bin_items:
        # Overlap check: two boxes overlap if they intersect in all 3 dimensions
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True


def _find_position(bin_items, w, h, d, W, H, D):
    """Find a valid position for a box in a bin.
    Uses a candidate-point strategy: try positions derived from existing items' faces
    and the origin, sorted by a heuristic (prefer low z, then low y, then low x)."""
    # Generate candidate x, y, z coordinates
    xs = [0]
    ys = [0]
    zs = [0]
    for (bx, by, bz, bw, bh, bd) in bin_items:
        xs.append(bx + bw)
        ys.append(by + bh)
        zs.append(bz + bd)
        # Also try positions adjacent to items in each axis
        if bx >= w:
            xs.append(bx - w)
        if by >= h:
            ys.append(by - h)
        if bz >= d:
            zs.append(bz - d)

    xs = sorted(set(x for x in xs if 0 <= x <= W - w))
    ys = sorted(set(y for y in ys if 0 <= y <= H - h))
    zs = sorted(set(z for z in zs if 0 <= z <= D - d))

    # If no candidates, try the origin
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    if not zs:
        zs = [0]

    # Try all combinations, sorted by (z, y, x) to prefer "lowest" placement
    candidates = []
    for z in zs:
        for y in ys:
            for x in xs:
                candidates.append((z, y, x))
    candidates.sort()

    for (z, y, x) in candidates:
        if _can_place(bin_items, x, y, z, w, h, d, W, H, D):
            return (x, y, z)
    return None


def _pack_greedy(boxes, container, sort_key):
    """Greedy packing: sort boxes by sort_key, place each in the first bin where it fits.
    Returns a list of (bin, x, y, z, w, h, d) tuples."""
    W, H, D = container
    n = len(boxes)
    # Sort indices by the key
    indexed = sorted(range(n), key=lambda i: sort_key(boxes[i]), reverse=True)

    bins = []  # list of lists of (x, y, z, w, h, d)
    placements = [None] * n

    for idx in indexed:
        w, h, d = boxes[idx]
        placed = False
        # Try existing bins
        for bi in range(len(bins)):
            pos = _find_position(bins[bi], w, h, d, W, H, D)
            if pos is not None:
                x, y, z = pos
                bins[bi].append((x, y, z, w, h, d))
                placements[idx] = (bi, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            # Start a new bin
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[idx] = (bi, 0, 0, 0, w, h, d)

    return placements, bins


def _score_placements(placements, container):
    """Compute the score: mean B + 0.1*(1 - last-bin occupied volume / container volume).
    B = number of occupied bins (max bin label + 1)."""
    if not placements:
        return 0.0
    W, H, D = container
    cont_vol = W * H * D
    max_bin = max(p[0] for p in placements)
    B = max_bin + 1

    # Compute occupied volume in the last bin
    last_bin_vol = 0
    for p in placements:
        if p[0] == max_bin:
            last_bin_vol += p[4] * p[5] * p[6]

    frac = last_bin_vol / cont_vol if cont_vol > 0 else 0.0
    return B + 0.1 * (1.0 - frac)


def _local_search(placements, bins, boxes, container, max_iters=200):
    """Try to improve by moving boxes from higher bins to lower bins, or within a bin.
    Returns improved placements and bins if better."""
    W, H, D = container
    current_score = _score_placements(placements, container)
    best_score = current_score
    best_placements = [p[:] for p in placements]
    best_bins = [list(b) for b in bins]

    for _ in range(max_iters):
        improved = False
        # For each box, try moving it to a lower bin
        for idx in range(len(boxes)):
            p = placements[idx]
            cur_bin, x, y, z, w, h, d = p
            if cur_bin == 0:
                continue
            # Try to remove from current bin and place in a lower bin
            # First remove from current bin
            bins[cur_bin].remove((x, y, z, w, h, d))
            # Try bins 0..cur_bin-1
            found = False
            for bi in range(cur_bin):
                pos = _find_position(bins[bi], w, h, d, W, H, D)
                if pos is not None:
                    nx, ny, nz = pos
                    bins[bi].append((nx, ny, nz, w, h, d))
                    placements[idx] = (bi, nx, ny, nz, w, h, d)
                    new_score = _score_placements(placements, container)
                    if new_score < best_score:
                        best_score = new_score
                        best_placements = [q[:] for q in placements]
                        best_bins = [list(b) for b in bins]
                    else:
                        # Revert
                        bins[bi].remove((nx, ny, nz, w, h, d))
                        bins[cur_bin].append((x, y, z, w, h, d))
                        placements[idx] = (cur_bin, x, y, z, w, h, d)
                    found = True
                    break
            if not found:
                # Put back in original bin
                bins[cur_bin].append((x, y, z, w, h, d))
                placements[idx] = (cur_bin, x, y, z, w, h, d)

        # Also try repositioning within the same bin
        for idx in range(len(boxes)):
            p = placements[idx]
            cur_bin, x, y, z, w, h, d = p
            # Remove
            bins[cur_bin].remove((x, y, z, w, h, d))
            pos = _find_position(bins[cur_bin], w, h, d, W, H, D)
            if pos is not None:
                nx, ny, nz = pos
                bins[cur_bin].append((nx, ny, nz, w, h, d))
                placements[idx] = (cur_bin, nx, ny, nz, w, h, d)
                new_score = _score_placements(placements, container)
                if new_score < best_score:
                    best_score = new_score
                    best_placements = [q[:] for q in placements]
                    best_bins = [list(b) for b in bins]
                else:
                    bins[cur_bin].remove((nx, ny, nz, w, h, d))
                    bins[cur_bin].append((x, y, z, w, h, d))
                    placements[idx] = (cur_bin, x, y, z, w, h, d)
            else:
                bins[cur_bin].append((x, y, z, w, h, d))
                placements[idx] = (cur_bin, x, y, z, w, h, d)

        if best_score < current_score:
            current_score = best_score
        else:
            break

    return best_placements, best_bins, best_score


def pack(boxes, container):
    """Main entry point. Uses multi-strategy greedy with local search."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    # Define multiple sorting strategies
    strategies = [
        ("vol_desc", lambda b: -(b[0]*b[1]*b[2])),
        ("maxdim_desc", lambda b: -(max(b))),
        ("sumdim_desc", lambda b: -(b[0]+b[1]+b[2])),
        ("vol_asc", lambda b: b[0]*b[1]*b[2]),
        ("maxdim_asc", lambda b: max(b)),
        ("sumdim_asc", lambda b: b[0]+b[1]+b[2]),
    ]

    best_placements = None
    best_score = float('inf')

    for name, sort_key in strategies:
        if time.time() > deadline:
            break
        placements, bins = _pack_greedy(boxes, container, sort_key)
        if placements is None:
            continue
        score = _score_placements(placements, container)
        if score < best_score:
            best_score = score
            best_placements = [p[:] for p in placements]

        # Local search on the best so far
        if time.time() < deadline - 5.0:
            improved_pl, improved_bins, improved_score = _local_search(
                placements, bins, boxes, container, max_iters=50
            )
            if improved_score < best_score:
                best_score = improved_score
                best_placements = [p[:] for p in improved_pl]

    if best_placements is None:
        # Fallback: simple shelf packing
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
        best_placements = placements

    return best_placements

# EVOLVE-BLOCK-END
