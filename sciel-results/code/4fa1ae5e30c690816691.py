"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = None

def _now():
    return time.time()

def _collinear3(a, b, c):
    (x1, y1), (x2, y2), (x3, y3) = a, b, c
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _has_four_collinear(pts):
    n = len(pts)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(dx, dy)
            if g == 0:
                continue
            dx //= g
            dy //= g
            key = (x1, y1, dx, dy)
            if key in seen:
                return True
            seen.add(key)
    return False

def _gcd(a, b):
    a = abs(a); b = abs(b)
    while b:
        a, b = b, a % b
    return a

def _build_triple_index(pts):
    """Return list of triples (i,j,k) that are collinear."""
    n = len(pts)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(dx, dy)
            if g == 0:
                continue
            dx //= g
            dy //= g
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                if (x3 - x1) * dy - (y3 - y1) * dx == 0:
                    triples.append((i, j, k))
    return triples

def _independence_number(pts):
    """Exact largest subset with no three collinear via branch and bound."""
    n = len(pts)
    triples = _build_triple_index(pts)
    if not triples:
        return n
    # conflict: for each point, bitmask of points that form a collinear triple with it AND some third
    # We'll use a recursive search: include/exclude with pruning.
    # Represent candidate set as a bitset (int).
    # Precompute for each point p, the set of points q such that there exists r with (p,q,r) collinear.
    # Actually we need: a set S is valid iff it contains no triple from `triples`.
    # Branch and bound: try to build a maximal valid set.
    # Heuristic: order points, use lower bound from greedy.

    # Build adjacency of "conflict": p and q conflict if there is r with (p,q,r) collinear.
    conflict = [0] * n
    for (i, j, k) in triples:
        conflict[i] |= (1 << j) | (1 << k)
        conflict[j] |= (1 << i) | (1 << k)
        conflict[k] |= (1 << i) | (1 << j)

    # Greedy lower bound for initial best
    def greedy_lb(cand):
        # cand is a set of indices
        chosen = []
        remaining = list(cand)
        random.shuffle(remaining)
        for p in remaining:
            ok = True
            for q in chosen:
                if (conflict[p] >> q) & 1:
                    ok = False
                    break
            if ok:
                chosen.append(p)
        return len(chosen)

    best = [0]
    full_mask = (1 << n) - 1

    # Sort points by degree for branching
    order = sorted(range(n), key=lambda p: bin(conflict[p]).count('1'), reverse=True)

    import sys
    sys.setrecursionlimit(10000)

    def search(available, chosen_count, chosen_set):
        if _now() > _DEADLINE - 5:
            return
        if available == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Upper bound: chosen_count + popcount(available)
        ub = chosen_count + bin(available).count('1')
        if ub <= best[0]:
            return
        # Pick a point to branch on: highest degree within available
        # Try to find a point with many conflicts in available
        best_p = -1
        best_conf = -1
        tmp = available
        while tmp:
            lsb = tmp & (-tmp)
            p = lsb.bit_length() - 1
            c = bin(conflict[p] & available).count('1')
            if c > best_conf:
                best_conf = c
                best_p = p
            tmp ^= lsb
        if best_p < 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Branch 1: include best_p
        new_avail = available & ~(1 << best_p)
        # Remove points that conflict with best_p
        new_avail &= ~conflict[best_p]
        search(new_avail, chosen_count + 1, chosen_set | (1 << best_p))
        # Branch 2: exclude best_p
        search(available & ~(1 << best_p), chosen_count, chosen_set)

    search(full_mask, 0, 0)
    return best[0]

def _local_search(pts, time_limit):
    """Try to reduce independence number by swapping points."""
    pts = list(pts)
    current_score = _independence_number(pts)
    best_score = current_score
    best_pts = list(pts)
    end = time_limit
    # Candidate pool: grid points
    pool = []
    for x in range(-30, 31):
        for y in range(-30, 31):
            pool.append((x, y))
    random.shuffle(pool)
    while _now() < end - 5:
        improved = False
        for trial in range(200):
            if _now() > end - 5:
                break
            # Pick a random point to remove and a random pool point to add
            idx = random.randrange(len(pts))
            new_pts = pts[:idx] + pts[idx+1:]
            # Try a few pool candidates
            for cand in pool[trial % len(pool) : trial % len(pool) + 3]:
                if cand in new_pts:
                    continue
                trial_pts = new_pts + [cand]
                if len(set(trial_pts)) != 20:
                    continue
                if _has_four_collinear(trial_pts):
                    continue
                # Quick check: compute score
                s = _independence_number(trial_pts)
                if s < best_score:
                    best_score = s
                    best_pts = list(trial_pts)
                    pts = list(trial_pts)
                    current_score = s
                    improved = True
                    break
        if not improved and best_score <= 11:
            break
    return best_pts, best_score

def _structured_seed():
    """Build a seed with many collinear triples from intersecting short lines."""
    # Use a grid of lines: horizontal, vertical, and diagonal lines of length 3-4
    # Place points at integer coordinates
    pts = set()
    # Several horizontal lines
    for y in [-6, -3, 0, 3, 6]:
        for x in [-6, -3, 0]:
            pts.add((x, y))
    # Several vertical lines
    for x in [3, 6]:
        for y in [-6, -3, 0, 3, 6]:
            pts.add((x, y))
    # Diagonals
    for d in [-3, 0, 3]:
        for x in [-6, -3, 0, 3]:
            y = x + d
            if -6 <= y <= 6:
                pts.add((x, y))
    # Convert to list, take first 20 distinct
    pts = list(pts)
    random.shuffle(pts)
    return pts[:20]

def construct():
    global _DEADLINE
    _DEADLINE = _now() + 140  # safety margin before 150s kill
    random.seed(42)

    # Start from structured seed
    pts = _structured_seed()
    # Ensure validity
    if _has_four_collinear(pts):
        # Fallback to a safer seed
        pts = []
        for x in range(-10, 10):
            pts.append([x, x * x])
        pts = [tuple(p) for p in pts]

    pts, score = _local_search(pts, _DEADLINE)

    # Final validation
    if len(set(pts)) != 20:
        pts = [(x, x * x) for x in range(-10, 10)]
    if _has_four_collinear(pts):
        pts = [(x, x * x) for x in range(-10, 10)]
    for p in pts:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            pts = [(x, x * x) for x in range(-10, 10)]
            break

    return [[x, y] for (x, y) in pts]

# EVOLVE-BLOCK-END
