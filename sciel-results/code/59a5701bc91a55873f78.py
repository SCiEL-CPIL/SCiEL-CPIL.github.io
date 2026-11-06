"""Improved program: randomized greedy with bitmask conflict precomputation."""
import time
import random

N = 16
TOTAL = N * N
DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _precompute_conflicts():
    """pair_conflict[a][b] = bitmask of points c where {a,b,c} is isosceles."""
    pair_conflict = [[0] * TOTAL for _ in range(TOTAL)]
    pts = [(x, y) for y in range(N) for x in range(N)]
    for a in range(TOTAL):
        ax, ay = pts[a]
        for b in range(a + 1, TOTAL):
            bx, by = pts[b]
            d_ab = (ax - bx) ** 2 + (ay - by) ** 2
            mask = 0
            for c in range(TOTAL):
                if c == a or c == b:
                    continue
                cx, cy = pts[c]
                d_ac = (ax - cx) ** 2 + (ay - cy) ** 2
                d_bc = (bx - cx) ** 2 + (by - cy) ** 2
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    mask |= (1 << c)
            pair_conflict[a][b] = mask
            pair_conflict[b][a] = mask
    return pair_conflict, pts


def _randomized_greedy(pair_conflict, pts, rng):
    """Pick random unblocked points until none remain. Returns list of point indices."""
    blocked = 0
    chosen = []
    remaining = list(range(TOTAL))
    
    while remaining:
        # Find an unblocked point in remaining
        found = -1
        for i in range(len(remaining)):
            p = remaining[i]
            if not (blocked >> p) & 1:
                found = i
                break
        if found == -1:
            break
        p = remaining[found]
        # Remove from remaining
        remaining[found] = remaining[-1]
        remaining.pop()
        chosen.append(p)
        # Update blocked with conflicts of p against all previously chosen
        for q in chosen[:-1]:
            blocked |= pair_conflict[p][q]
    
    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + (330.0 - DEADLINE_MARGIN)

    pair_conflict, pts = _precompute_conflicts()

    best_set = None
    best_size = float('inf')

    rng = random.Random(12345)

    while time.time() < deadline:
        chosen = _randomized_greedy(pair_conflict, pts, rng)
        size = len(chosen)
        if size < best_size:
            best_size = size
            best_set = list(chosen)
            if best_size <= 6:
                break

    if best_set is None:
        chosen = _randomized_greedy(pair_conflict, pts, random.Random(0))
        best_set = chosen

    return [pts[i] for i in best_set]
