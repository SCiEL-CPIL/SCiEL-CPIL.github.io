"""
Combined approach: structured seeds + fast local search with two-point swaps
and a compatibility-graph-inspired global search for exact-4 lines.
"""
import time
import math
import random
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 115.0
_BOUND = 1000


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _score_full(points):
    """Returns (num_exact4, num_5plus) for a set of 16 points."""
    n = len(points)
    counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    n4 = n5 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
    return n4, n5


def _valid(points):
    """True iff no five are collinear."""
    n = len(points)
    counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= 10:
                return False
    return True


def _find_exact4_lines(points):
    """Return list of lines (as keys) that contain exactly 4 points."""
    n = len(points)
    counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    lines4 = []
    for k, c in counts.items():
        if c == 6:
            lines4.append(k)
    return lines4


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_perturbed():
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_affine():
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_hexagonal():
    pts = [[0, 0]]
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]:
        pts.append([dx, dy])
    ring2 = [(2,0),(-2,0),(0,2),(0,-2),(2,1),(-2,-1),(1,2),(-1,-2)]
    for p in ring2:
        pts.append(list(p))
    return pts[:16]


def _local_search(points, rng, time_budget):
    """Local search with single-point moves and two-point swaps."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5 = _score_full(cur)
    best_key = (cur_n4, -cur_n5)
    best_pts = [list(p) for p in cur]
    
    while time.time() < time_budget:
        improved = False
        
        # --- Single-point moves ---
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            w = 5
            
            best_local = None
            best_local_key = None
            
            for dx in range(-w, w + 1):
                if time.time() >= time_budget:
                    break
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue
                    
                    cand = [list(p) for p in cur]
                    cand[i] = np_
                    
                    if not _valid(cand):
                        continue
                    
                    n4, n5 = _score_full(cand)
                    key = (n4, -n5)
                    if best_local_key is None or key > best_local_key:
                        best_local_key = key
                        best_local = np_
            
            if best_local is not None:
                cur_key = (cur_n4, -cur_n5)
                if best_local_key > cur_key:
                    cur[i] = best_local
                    cur_n4, cur_n5 = best_local_key[0], -best_local_key[1]
                    improved = True
                    if (cur_n4, -cur_n5) > best_key:
                        best_key = (cur_n4, -cur_n5)
                        best_pts = [list(p) for p in cur]
        
        # --- Two-point swaps (structural change) ---
        if time.time() < time_budget:
            for _ in range(50):  # limited swaps per iteration
                if time.time() >= time_budget:
                    break
                i, j = rng.sample(range(n), 2)
                x0, y0 = cur[i]
                x1, y1 = cur[j]
                
                # Try swapping positions
                cand = [list(p) for p in cur]
                cand[i] = [x1, y1]
                cand[j] = [x0, y0]
                
                if len(set(tuple(p) for p in cand)) != 16:
                    continue
                if not _valid(cand):
                    continue
                
                n4, n5 = _score_full(cand)
                key = (n4, -n5)
                cur_key = (cur_n4, -cur_n5)
                if key > cur_key:
                    cur = cand
                    cur_n4, cur_n5 = n4, n5
                    improved = True
                    if (cur_n4, -cur_n5) > best_key:
                        best_key = (cur_n4, -cur_n5)
                        best_pts = [list(p) for p in cur]
        
        # --- Random kick to escape local optimum ---
        if not improved:
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(200):
                if time.time() >= time_budget:
                    break
                nx = x0 + rng.randint(-8, 8)
                ny = y0 + rng.randint(-8, 8)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue
                cand = [list(p) for p in cur]
                cand[i] = np_
                if not _valid(cand):
                    continue
                n4, n5 = _score_full(cand)
                key = (n4, -n5)
                cur_key = (cur_n4, -cur_n5)
                if key >= cur_key:
                    cur[i] = np_
                    cur_n4, cur_n5 = n4, n5
                    if (cur_n4, -cur_n5) > best_key:
                        best_key = (cur_n4, -cur_n5)
                        best_pts = [list(p) for p in cur]
    
    return best_pts, best_key


def construct():
    rng = random.Random(42)
    
    seeds = [
        _seed_grid(),
        _seed_perturbed(),
        _seed_affine(),
        _seed_staggered(),
        _seed_hexagonal(),
    ]
    
    best_overall = None
    best_key_overall = None
    
    for seed in seeds:
        if time.time() >= _DEADLINE:
            break
        if not _valid(seed):
            continue
        pts, key = _local_search(seed, rng, _DEADLINE)
        if best_key_overall is None or key > best_key_overall:
            best_key_overall = key
            best_overall = pts
    
    # Final verification
    if best_overall is None:
        best_overall = _seed_grid()
    
    assert len(set(tuple(p) for p in best_overall)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in best_overall)
    assert not _has_five_collinear(best_overall)
    
    return [[int(p[0]), int(p[1])] for p in best_overall]


def _has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    if n < 5:
        return False
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    for key, pair_count in line_counts.items():
        k = int((1 + (1 + 8 * pair_count) ** 0.5) / 2)
        if k * (k - 1) // 2 == pair_count and k >= 5:
            return True
    return False
