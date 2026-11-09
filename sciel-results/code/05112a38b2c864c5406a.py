"""3D Container Loading - improved packing algorithm."""
import time
import random
import itertools
import numpy as np

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits_in_container(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    """Check if box at pos with box_dims overlaps any placed box."""
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    """Generate candidate positions from extreme points of placed boxes."""
    W, H, D = container
    candidates = {(0, 0, 0)}
    for (px, py, pz, pw, ph, pd) in placed:
        # Points at each face of the placed box
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        # Also corners
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
    # Filter valid candidates
    valid = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    return valid

def _score_position(pos, box_dims, placed, container):
    """Score a position: prefer compact (close to origin) and tight fits."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    # Prefer positions closer to origin (compact)
    dist = x + y + z
    # Prefer tighter fits (less wasted space along each axis)
    slack = (W - x - w) + (H - y - h) + (D - z - d)
    return dist * 0.5 + slack * 0.1

def _greedy_place(boxes, container, order, time_limit, start_time):
    """
    Greedy placement: for each box in given order, try all orientations and
    candidate positions, pick the best valid one.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    results = {}  # box_index -> (x, y, z, w, h, d) or None

    for idx in order:
        w, h, d = boxes[idx]
        best_pos = None
        best_orient = None
        best_score = float('inf')

        orients = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)

        for ow, oh, od in orients:
            for (cx, cy, cz) in candidates:
                if not _fits_in_container((ow, oh, od), (cx, cy, cz), container):
                    continue
                if not _no_overlap((ow, oh, od), (cx, cy, cz), placed):
                    continue
                s = _score_position((cx, cy, cz), (ow, oh, od), placed, container)
                if s < best_score:
                    best_score = s
                    best_pos = (cx, cy, cz)
                    best_orient = (ow, oh, od)

        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_orient
            placed.append((bx, by, bz, bw, bh, bd))
            results[idx] = (bx, by, bz, bw, bh, bd)
        else:
            results[idx] = None

        # Time check
        if time.time() - start_time > time_limit:
            break

    return placed, results

def _multi_start(boxes, container, time_limit):
    """
    Try multiple random orderings of boxes, keep the best packing.
    """
    W, H, D = container
    n = len(boxes)
    indices = list(range(n))

    # Precompute box volumes for sorting heuristics
    volumes = [w * h * d for w, h, d in boxes]

    # Sort strategies
    strategies = []

    # 1. Largest volume first
    strategies.append(sorted(indices, key=lambda i: volumes[i], reverse=True))

    # 2. Smallest volume first
    strategies.append(sorted(indices, key=lambda i: volumes[i]))

    # 3. Largest dimension first
    strategies.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))

    # 4. Smallest dimension first
    strategies.append(sorted(indices, key=lambda i: min(boxes[i])))

    # 5. Aspect ratio (elongated first)
    strategies.append(sorted(indices, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), reverse=True))

    # 6-10. Random shuffles
    for _ in range(5):
        shuffled = indices[:]
        random.shuffle(shuffled)
        strategies.append(shuffled)

    best_results = None
    best_volume = 0

    n_strats = len(strategies)
    time_per_strat = time_limit / n_strats

    for strat_idx, order in enumerate(strategies):
        start_time = time.time()
        placed, results = _greedy_place(boxes, container, order, time_per_strat, start_time)

        total_vol = sum(w * h * d for (x, y, z, w, h, d) in placed)
        if total_vol > best_volume:
            best_volume = total_vol
            best_results = results

        # If time is running out, stop
        if time.time() - start_time > time_limit:
            break

    if best_results is None:
        # Fallback: empty placement
        best_results = {i: None for i in range(n)}

    return best_results

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d)
    where bin=0 for loaded, bin=-1 for left out.
    """
    start_time = time.time()
    time_limit = 110.0  # safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    # Seed random for reproducibility within a run
    random.seed(42)

    # Multi-start greedy with different orderings
    results = _multi_start(boxes, container, time_limit)

    # Build output
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        if results.get(i) is not None:
            bx, by, bz, bw, bh, bd = results[i]
            placements.append((0, bx, by, bz, bw, bh, bd))
        else:
            placements.append((-1, 0, 0, 0, w, h, d))

    return placements
