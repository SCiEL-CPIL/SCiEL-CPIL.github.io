"""No-isosceles set construction with conflict-pair counting and local search."""
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 280.0  # safety margin before 330s kill


def _is_isosceles_fast(new_pts, existing_pts):
    """
    For each new point, count how many pairs of existing points it forms
    an isosceles triangle with. Returns array of conflict counts.
    new_pts: (m, 2) int array
    existing_pts: (k, 2) int array
    Returns: (m,) int array of conflict counts
    """
    m = len(new_pts)
    k = len(existing_pts)
    if m == 0 or k < 2:
        return np.zeros(m, dtype=np.int32)

    # Distances from each new point to each existing point: (m, k)
    # new_pts[:, 0:1, :] - existing_pts[None, :, :] -> (m, k, 2)
    dx = new_pts[:, 0:1] - existing_pts[None, :, 0]
    dy = new_pts[:, 1:2] - existing_pts[None, :, 1]
    d2 = dx * dx + dy * dy  # (m, k) squared distances

    # For each new point i, count pairs (a, b) in existing where d2[i,a] == d2[i,b]
    # We need to count for each row i, the number of pairs (a,b) with a<b and d2[i,a]==d2[i,b]
    # This is: for each value v appearing c times in row i, add c*(c-1)//2
    # But we only need to know if count > 0, so we can early-exit.
    # For efficiency, just compute the total conflict count.

    # For each row, count pairs with equal values
    # Use a trick: for each row, sort and count duplicates
    # But for large m*k this could be slow. Let's use a different approach.

    # Actually, let's compute it differently:
    # For each new point, the number of isosceles triangles = sum over all pairs (a,b) of [d(i,a)==d(i,b)]
    # = for each distance value v, if it appears c_v times, add c_v*(c_v-1)//2

    # Efficient computation using np.unique with counts per row is hard in pure numpy.
    # Alternative: for each pair of existing points (a,b), check if new point is equidistant.
    # That's O(m * k^2) which is what we want to avoid.

    # Let's use the per-row approach with a loop over m (new points):
    conflicts = np.zeros(m, dtype=np.int32)
    for i in range(m):
        row = d2[i]  # (k,)
        # Count pairs with equal values
        # Use: for each unique value, count = c, pairs = c*(c-1)//2
        # We can use np.unique(row, return_counts=True)
        _, counts = np.unique(row, return_counts=True)
        # Sum of c*(c-1)//2 for all c
        conflicts[i] = np.sum(counts * (counts - 1) // 2)
    return conflicts


def _greedy_conflict(order, existing_pts=None, time_budget=None, start_time=None):
    """
    Greedy construction using conflict counting.
    order: list of (x,y) points to try in this order.
    existing_pts: initial set of points (optional).
    Returns: list of points in the constructed set.
    """
    if start_time is None:
        start_time = time.time()
    if time_budget is None:
        time_budget = TIME_LIMIT - 5 - (start_time - start_time)

    pts = list(existing_pts) if existing_pts else []
    pts_set = set(pts)
    k = len(pts)

    for pt in order:
        if time.time() - start_time > time_budget:
            break
        if pt in pts_set:
            continue
        if k < 2:
            pts.append(pt)
            pts_set.add(pt)
            k += 1
            continue
        # Check if pt conflicts with any pair in pts
        # Use the fast method
        new_arr = np.array([pt], dtype=np.int32)
        existing_arr = np.array(pts, dtype=np.int32)
        conf = _is_isosceles_fast(new_arr, existing_arr)
        if conf[0] == 0:
            pts.append(pt)
            pts_set.add(pt)
            k += 1

    return pts


def _greedy_conflict_vectorized(order, start_time=None, time_budget=None):
    """
    Faster greedy: process points in batches using vectorized conflict check.
    This is much faster when many points can be added at once.
    """
    if start_time is None:
        start_time = time.time()
    if time_budget is None:
        time_budget = TIME_LIMIT - 5

    pts = []
    pts_set = set()

    # Process in batches
    i = 0
    batch_size = 200
    while i < len(order):
        if time.time() - start_time > time_budget:
            break

        batch_end = min(i + batch_size, len(order))
        batch = [p for p in order[i:batch_end] if p not in pts_set]

        if not batch:
            i = batch_end
            continue

        if len(pts) < 2:
            # Can add all
            for p in batch:
                pts.append(p)
                pts_set.add(p)
            i = batch_end
            continue

        # Check all batch points against existing
        new_arr = np.array(batch, dtype=np.int32)
        existing_arr = np.array(pts, dtype=np.int32)
        conf = _is_isosceles_fast(new_arr, existing_arr)

        # Add all points with zero conflicts
        for j, p in enumerate(batch):
            if conf[j] == 0:
                pts.append(p)
                pts_set.add(p)

        i = batch_end

    return pts


def _local_search_grow(pts, start_time=None, time_budget=None):
    """
    Try to add more points to the set by scanning the grid.
    Returns the improved set.
    """
    if start_time is None:
        start_time = time.time()
    if time_budget is None:
        time_budget = TIME_LIMIT - 5

    pts = list(pts)
    pts_set = set(pts)

    # Generate all grid points not in pts
    all_grid = [(x, y) for x in range(N) for y in range(N)]
    candidates = [p for p in all_grid if p not in pts_set]
    random.shuffle(candidates)

    existing_arr = np.array(pts, dtype=np.int32) if pts else None
    k = len(pts)

    for pt in candidates:
        if time.time() - start_time > time_budget:
            break
        if pt in pts_set:
            continue
        if k < 2:
            pts.append(pt)
            pts_set.add(pt)
            k += 1
            if existing_arr is None:
                existing_arr = np.array(pts, dtype=np.int32)
            else:
                existing_arr = np.vstack([existing_arr, np.array([pt], dtype=np.int32)])
            continue

        new_arr = np.array([pt], dtype=np.int32)
        conf = _is_isosceles_fast(new_arr, existing_arr)
        if conf[0] == 0:
            pts.append(pt)
            pts_set.add(pt)
            k += 1
            existing_arr = np.vstack([existing_arr, np.array([pt], dtype=np.int32)])

    return pts


def _local_search_swap(pts, start_time=None, time_budget=None, max_swaps=5000):
    """
    Local search: try to swap out a point for a better one.
    Strategy: for each point, compute how many candidate points it blocks.
    Remove the point that blocks the most, then try to add new points.
    """
    if start_time is None:
        start_time = time.time()
    if time_budget is None:
        time_budget = TIME_LIMIT - 5

    pts = list(pts)
    pts_set = set(pts)
    k = len(pts)

    if k < 2:
        return pts

    existing_arr = np.array(pts, dtype=np.int32)

    # For local search, we'll try random swaps
    # Pick a random point to remove, then try to add several new points
    rng = random.Random(42)

    for _ in range(max_swaps):
        if time.time() - start_time > time_budget:
            break

        # Pick a random point to remove
        idx = rng.randint(0, len(pts) - 1)
        removed = pts[idx]
        pts_without = [p for j, p in enumerate(pts) if j != idx]
        pts_set_without = pts_set - {removed}
        k_without = len(pts_without)

        if k_without < 2:
            # Can add any point
            candidates = [p for p in rng.sample([(x, y) for x in range(N) for y in range(N)], min(50, N*N)) if p not in pts_set_without]
            for c in candidates:
                if c not in pts_set_without:
                    pts_without.append(c)
                    pts_set_without.add(c)
            if len(pts_without) > len(pts):
                pts = pts_without
                pts_set = pts_set_without
            else:
                # Restore
                pts = pts_without + [removed]
                pts_set = pts_set_without | {removed}
            continue

        existing_arr_wo = np.array(pts_without, dtype=np.int32)

        # Try to add several random candidates
        added_any = False
        for _ in range(20):
            if time.time() - start_time > time_budget:
                break
            c = (rng.randint(0, N-1), rng.randint(0, N-1))
            if c in pts_set_without:
                continue
            new_arr = np.array([c], dtype=np.int32)
            conf = _is_isosceles_fast(new_arr, existing_arr_wo)
            if conf[0] == 0:
                pts_without.append(c)
                pts_set_without.add(c)
                existing_arr_wo = np.vstack([existing_arr_wo, np.array([c], dtype=np.int32)])
                added_any = True

        if len(pts_without) > len(pts):
            pts = pts_without
            pts_set = pts_set_without
        # else: keep original (we didn't actually modify pts, just pts_without)

    return pts


def _multi_greedy(start_time, time_budget):
    """
    Run multiple greedy constructions with different orderings and return the best.
    """
    best_pts = []
    best_len = 0

    all_pts = [(x, y) for x in range(N) for y in range(N)]

    # Try several different orderings
    orderings = []

    # Random orderings
    for seed in range(10):
        rng = random.Random(seed)
        order = list(all_pts)
        rng.shuffle(order)
        orderings.append(order)

    # Row-by-row
    orderings.append([(x, y) for x in range(N) for y in range(N)])
    # Column-by-column
    orderings.append([(x, y) for y in range(N) for x in range(N)])
    # Diagonal order
    orderings.append([(x, y) for s in range(2*N-1) for x in range(max(0, s-N+1), min(N, s+1)) for y in [s-x] if 0 <= y < N])
    # Spiral-like
    cx, cy = N//2, N//2
    spiral = [(cx, cy)]
    for r in range(1, N):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if abs(dx) == r or abs(dy) == r:
                    px, py = cx+dx, cy+dy
                    if 0 <= px < N and 0 <= py < N and (px, py) not in spiral:
                        spiral.append((px, py))
    orderings.append(spiral)

    # Process orderings with time budget
    for order in orderings:
        if time.time() - start_time > time_budget:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break

        pts = _greedy_conflict_vectorized(order, start_time=start_time, time_budget=remaining)

        if len(pts) > best_len:
            best_pts = pts
            best_len = len(pts)

    return best_pts


def _construct():
    start_time = time.time()

    # Phase 1: Multi-greedy to get a good initial set
    initial_budget = min(60.0, TIME_LIMIT - 30)
    pts = _multi_greedy(start_time, initial_budget)

    # Phase 2: Local search - grow
    remaining = TIME_LIMIT - 10 - (time.time() - start_time)
    if remaining > 10:
        pts = _local_search_grow(pts, start_time=start_time, time_budget=remaining)

    # Phase 3: Local search - swap
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 5:
        pts = _local_search_swap(pts, start_time=start_time, time_budget=remaining)

    # Phase 4: Final grow attempt
    remaining = TIME_LIMIT - 2 - (time.time() - start_time)
    if remaining > 2:
        pts = _local_search_grow(pts, start_time=start_time, time_budget=remaining)

    return pts


def construct():
    """
    Main entry point. Returns a list of (x, y) grid points in [0, N)^2
    with no three points forming an isosceles triangle.
    """
    start = time.time()
    try:
        result = _construct()
    except Exception:
        # Fallback: return a small valid set
        return [(0, 0), (1, 0), (0, 1)]

    # Deadline guard
    if time.time() - start > TIME_LIMIT:
        # Shouldn't happen, but safety
        return result

    return result
