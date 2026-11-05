"""Rewritten no-isosceles set constructor: fast greedy + remove-and-refill local search."""
import time
import random

N = 64
DEADLINE = time.time() + 295.0  # safety margin before the 330s kill

def _dist_set(points):
    """Build per-point set of squared distances to every other point."""
    ds = [set() for _ in points]
    m = len(points)
    for i in range(m):
        xi, yi = points[i]
        di = ds[i]
        for j in range(i + 1, m):
            xj, yj = points[j]
            d = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
            di.add(d)
            ds[j].add(d)
    return ds

def _can_add(points, ds, x, y):
    """Return True if (x,y) can be added without creating an isosceles triangle."""
    for (px, py), s in zip(points, ds):
        d = (x - px) * (x - px) + (y - py) * (y - py)
        if d in s:
            return False
    return True

def _try_add(points, ds, x, y):
    """Add (x,y) to points and update ds in place. Caller must ensure validity."""
    points.append((x, y))
    k = len(points) - 1
    new_set = set()
    for (px, py), s in zip(points[:-1], ds[:-1]):
        d = (x - px) * (x - px) + (y - py) * (y - py)
        new_set.add(d)
        s.add(d)
    ds.append(new_set)

def _greedy_build(rng):
    """Greedy build with a randomized priority order."""
    points = []
    ds = []
    # Randomized scan order over the grid
    order = list(range(N * N))
    rng.shuffle(order)
    for idx in order:
        x = idx % N
        y = idx // N
        if _can_add(points, ds, x, y):
            _try_add(points, ds, x, y)
    return points, ds

def _local_search(points, ds, rng, iters=6000):
    """Remove-and-refill local search. Returns improved (points, ds)."""
    n = N
    for _ in range(iters):
        if time.time() > DEADLINE:
            break
        m = len(points)
        if m == 0:
            break
        ri = rng.randrange(m)
        # Save state
        saved_points = points[:]
        saved_ds = [s.copy() for s in ds]
        rx, ry = points[ri]
        # Remove point ri
        for j in range(len(ds)):
            if j != ri:
                d = (rx - points[j][0]) ** 2 + (ry - points[j][1]) ** 2
                ds[j].discard(d)
        del points[ri]
        del ds[ri]
        # Try to refill with up to 3 points
        added = 0
        # Build candidate list: random grid points
        cands = []
        for _ in range(40):
            x = rng.randrange(n)
            y = rng.randrange(n)
            cands.append((x, y))
        # Also try near the removed point
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                cx = rx + dx
                cy = ry + dy
                if 0 <= cx < n and 0 <= cy < n:
                    cands.append((cx, cy))
        rng.shuffle(cands)
        seen = set()
        for (x, y) in cands:
            if (x, y) in seen:
                continue
            seen.add((x, y))
            if _can_add(points, ds, x, y):
                _try_add(points, ds, x, y)
                added += 1
                if added >= 3:
                    break
        # Keep if net gain > 0, else revert
        if len(points) <= m:
            points[:] = saved_points
            ds[:] = saved_ds
    return points, ds

def construct():
    rng = random.Random(12345)
    # Best set found so far
    best_points, best_ds = _greedy_build(rng)
    best_len = len(best_points)
    # Local search on the greedy result
    points, ds = _local_search(best_points, best_ds, rng)
    if len(points) > best_len:
        best_points, best_ds, best_len = points, ds, len(points)
    # If time remains, do more restarts with fresh random orders
    while time.time() < DEADLINE - 5.0:
        # Fresh greedy with new seed
        rng2 = random.Random(rng.randrange(1 << 30))
        p2, d2 = _greedy_build(rng2)
        if len(p2) > best_len:
            best_points, best_ds, best_len = p2, d2, len(p2)
        # Local search on the new result
        p2, d2 = _local_search(p2, d2, rng2, iters=3000)
        if len(p2) > best_len:
            best_points, best_ds, best_len = p2, d2, len(p2)
    return best_points

# EVOLVE-BLOCK-END
