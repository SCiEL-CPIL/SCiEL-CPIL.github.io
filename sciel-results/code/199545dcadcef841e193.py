import time
import random

def _extreme_points(placed, W, H, D):
    """Generate candidate positions: the 6 faces of each placed box extended to the origin,
    plus the origin itself. Deduplicate and keep only those inside the container."""
    pts = set()
    pts.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        # face on +x side
        pts.add((px + pw, py, pz))
        # face on +y side
        pts.add((px, py + ph, pz))
        # face on +z side
        pts.add((px, py, pz + pd))
        # corner combinations (these help fill gaps)
        pts.add((px + pw, py + ph, pz))
        pts.add((px + pw, py, pz + pd))
        pts.add((px, py + ph, pz + pd))
        pts.add((px + pw, py + ph, pz + pd))
    # Filter: must be inside container (position coords non-negative and within bounds)
    valid = []
    for (x, y, z) in pts:
        if x < 0 or y < 0 or z < 0:
            continue
        # We'll check bounds per-orientation later, but filter obviously out-of-bounds
        if x >= W or y >= H or z >= D:
            continue
        valid.append((x, y, z))
    return valid

def _fits(dim, pos, W, H, D):
    w, h, d = dim
    x, y, z = pos
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(dim, pos, placed):
    w, h, d = dim
    x, y, z = pos
    for (px, py, pz, pw, ph, pd) in placed:
        if x < px + pw and x + w > px and \
           y < py + ph and y + h > py and \
           z < pz + pd and z + d > pz:
            return False
    return True

def _orientations(w, h, d):
    """Return all 6 unique orientations."""
    seen = set()
    res = []
    for a in (w, h, d):
        for b in (w, h, d):
            for c in (w, h, d):
                if a == w or b == w or c == w:
                    pass
                # Actually just use permutations
    # Simpler: use itertools.permutations
    import itertools
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            res.append(perm)
    return res

def _greedy_place(boxes, order_idx, W, H, D, orient_strategy, deadline):
    """
    boxes: list of (w,h,d)
    order_idx: list of indices in placement order
    orient_strategy: 0=try all orientations pick first fit, 1=try all pick min x+y+z, 
                     2=prefer orientation with smallest max dim first
    Returns (placements, filled_volume)
    """
    placed = []  # (x,y,z,w,h,d)
    placements = [None] * len(boxes)  # will fill with (bin, x, y, z, w, h, d)
    filled = 0.0
    n = len(boxes)

    for idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        orients = _orientations(w, h, d)
        # Sort orientations by strategy
        if orient_strategy == 2:
            orients.sort(key=lambda o: max(o))
        else:
            # keep natural order
            pass

        best = None  # (x, y, z, orient_dim)
        # Generate extreme points from current placed boxes
        pts = _extreme_points(placed, W, H, D)
        # Sort points by proximity to origin for deterministic tie-breaking
        pts.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))

        found = False
        for (x, y, z) in pts:
            for dim in orients:
                ow, oh, od = dim
                if not _fits(dim, (x, y, z), W, H, D):
                    continue
                if not _no_overlap(dim, (x, y, z), placed):
                    continue
                # Feasible
                if orient_strategy == 1:
                    # pick min x+y+z; since pts sorted, first found is best
                    best = (x, y, z, dim)
                    found = True
                    break
                else:
                    best = (x, y, z, dim)
                    found = True
                    break
            if found:
                break

        if found:
            bx, by, bz, (bw, bh, bd) = best
            placed.append((bx, by, bz, bw, bh, bd))
            placements[idx] = (0, bx, by, bz, bw, bh, bd)
            filled += bw * bh * bd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any remaining None (due to deadline) as left out
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements, filled

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start = time.time()
    deadline = start + 100  # safety margin before 120s limit

    # Precompute volumes
    volumes = [w * h * d for (w, h, d) in boxes]
    max_dims = [max(w, h, d) for (w, h, d) in boxes]
    min_dims = [min(w, h, d) for (w, h, d) in boxes]
    footprints = [w * h for (w, h, d) in boxes]

    # Build multiple orderings
    orderings = []
    # 1. Largest volume first
    orderings.append(sorted(range(n), key=lambda i: -volumes[i]))
    # 2. Largest max dim first
    orderings.append(sorted(range(n), key=lambda i: -max_dims[i]))
    # 3. Largest footprint first
    orderings.append(sorted(range(n), key=lambda i: -footprints[i]))
    # 4. Smallest min dim first (fit small gaps)
    orderings.append(sorted(range(n), key=lambda i: min_dims[i]))
    # 5. Original order
    orderings.append(list(range(n)))
    # 6. Largest volume, ties broken by original index
    orderings.append(sorted(range(n), key=lambda i: (-volumes[i], i)))

    # Deduplicate orderings
    seen = set()
    unique_orderings = []
    for o in orderings:
        key = tuple(o)
        if key not in seen:
            seen.add(key)
            unique_orderings.append(o)

    orient_strategies = [0, 1, 2]

    best_placements = None
    best_filled = -1.0

    # Try each combination
    for oi, order in enumerate(unique_orderings):
        for strat in orient_strategies:
            if time.time() > deadline:
                break
            placements, filled = _greedy_place(boxes, order, W, H, D, strat, deadline)
            if filled > best_filled:
                best_filled = filled
                best_placements = placements
        if time.time() > deadline:
            break

    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    return best_placements
