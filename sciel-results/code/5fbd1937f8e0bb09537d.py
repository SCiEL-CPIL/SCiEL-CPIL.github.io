"""Search for 20 integer points with no four collinear and a small no-three-collinear subset."""
import time
import random
from math import gcd
from itertools import combinations

_DEADLINE_SECONDS = 135.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if found."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dxn, dyn = dx // g, dy // g
            else:
                dxn, dyn = dx, dy
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            cnt = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[k][0] - points[i][0]) * dyn == (points[k][1] - points[i][1]) * dxn:
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False


def _max_no_three_collinear(points, deadline=None):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    if n <= 2:
        return n

    collinear_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    collinear_triples.append((i, j, k))
    
    forbidden_pairs = [[] for _ in range(n)]
    for (i, j, k) in collinear_triples:
        forbidden_pairs[i].append((j, k))
        forbidden_pairs[j].append((i, k))
        forbidden_pairs[k].append((i, j))
    
    best = [0]
    
    def can_add(i, chosen):
        for (a, b) in forbidden_pairs[i]:
            if (chosen >> a) & 1 and (chosen >> b) & 1:
                return False
        return True
    
    def branch(start, size, chosen):
        if deadline is not None and time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        if size + (n - start) <= best[0]:
            return
        if can_add(start, chosen):
            branch(start + 1, size + 1, chosen | (1 << start))
        branch(start + 1, size, chosen)
    
    branch(0, 0, 0)
    return best[0]


def _construct_perturbed_grids(rng):
    """Construct two 3x3 grids, perturb them to break 4-collinearities."""
    base_grid = [[i, j] for i in range(3) for j in range(3)]
    
    def perturb_grid(grid, offset_x, offset_y, rng):
        pts = []
        for p in grid:
            px = p[0] + offset_x + rng.randint(-2, 2)
            py = p[1] + offset_y + rng.randint(-2, 2)
            pts.append([px, py])
        return pts

    for _ in range(1000):
        if _remaining() < 10.0:
            break
        
        # Two grids far apart
        g1 = perturb_grid(base_grid, 0, 0, rng)
        g2 = perturb_grid(base_grid, 20, 20, rng)
        
        pts = g1 + g2
        
        # Check distinct
        if len(set(map(tuple, pts))) < 20:
            continue
            
        # Check bounds
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
            
        # Check 4-collinear
        if _has_four_collinear(pts):
            continue
            
        return pts
    return None


def _local_search(points, rng, deadline=None):
    """Local search to minimize score."""
    pts = [p[:] for p in points]
    current_score = _max_no_three_collinear(pts, deadline=deadline)
    best_score = current_score
    best_pts = [p[:] for p in pts]
    
    n = len(pts)
    for _ in range(500):
        if deadline is not None and time.time() > deadline:
            break
        
        i = rng.randint(0, n - 1)
        orig_i = pts[i][:]
        
        for attempt in range(20):
            if deadline is not None and time.time() > deadline:
                break
            new_x = orig_i[0] + rng.randint(-5, 5)
            new_y = orig_i[1] + rng.randint(-5, 5)
            
            if (new_x, new_y) == (orig_i[0], orig_i[1]):
                continue
            if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(pts) if j != i):
                continue
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
                
            pts[i] = [new_x, new_y]
            
            if not _has_four_collinear(pts):
                s = _max_no_three_collinear(pts, deadline=deadline)
                if s < best_score:
                    best_score = s
                    best_pts = [p[:] for p in pts]
                    break
                else:
                    pts[i] = orig_i
            else:
                pts[i] = orig_i
        else:
            pts[i] = orig_i

    return best_pts, best_score


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    rng = random.Random(42)
    
    # Strategy 1: Perturbed Grids
    for attempt in range(100):
        if _remaining() < 20.0:
            break
        pts = _construct_perturbed_grids(rng)
        if pts is None:
            continue
        s = _max_no_three_collinear(pts, deadline=_START + _DEADLINE_SECONDS)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            if best_score <= 8:
                break
    
    # Strategy 2: Local Search
    if best_points is not None and _remaining() > 15.0:
        improved_pts, improved_score = _local_search(
            best_points, rng, deadline=_START + _DEADLINE_SECONDS - 5.0
        )
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Fallback
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
        
    return best_points
