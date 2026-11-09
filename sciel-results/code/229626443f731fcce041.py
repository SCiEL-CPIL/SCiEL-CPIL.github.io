"""Integral point sets in general position - search for maximum size."""
import time
from math import isqrt

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = pts[k]
                if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    x4, y4 = pts[l]
                    # Circle through (x1,y1),(x2,y2),(x3,y3)
                    # Check if (x4,y4) is on it
                    # Using determinant: |x y x^2+y^2 1| = 0
                    a1, b1, c1 = x1, y1, x1*x1 + y1*y1
                    a2, b2, c2 = x2, y2, x2*x2 + y2*y2
                    a3, b3, c3 = x3, y3, x3*x3 + y3*y3
                    a4, b4, c4 = x4, y4, x4*x4 + y4*y4
                    # det of 4x4 matrix
                    # |a b c 1|
                    # |a b c 1|
                    # |a b c 1|
                    # |a b c 1|
                    d = (a1*(b2*(c3*1 - 1*c4) - c2*(b3*1 - 1*b4) + 1*(b3*c4 - c3*b4))
                         - b1*(a2*(c3*1 - 1*c4) - c2*(a3*1 - 1*b4) + 1*(a3*c4 - c3*a4))
                         + c1*(a2*(b3*1 - 1*b4) - b2*(a3*1 - 1*b4) + 1*(a3*b4 - b3*a4))
                         - 1*(a2*(b3*c4 - c3*b4) - b2*(a3*c4 - c3*a4) + c2*(a3*b4 - b3*a4)))
                    if d == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx*dx + D*dy*dy
            if not _is_square(d2):
                return False
    return True

def _validate(pts, D):
    if len(pts) < 3:
        return False
    if len(set(pts)) != len(pts):
        return False
    if not _all_distances_rational(pts, D):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

def _try_add_point(pts, D, candidate):
    """Check if adding candidate to pts keeps validity."""
    for p in pts:
        dx = candidate[0] - p[0]
        dy = candidate[1] - p[1]
        d2 = dx*dx + D*dy*dy
        if not _is_square(d2):
            return False
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = candidate
            dx = x2 - x1
            dy = y2 - y1
            if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                x4, y4 = candidate
                a1, b1, c1 = x1, y1, x1*x1 + y1*y1
                a2, b2, c2 = x2, y2, x2*x2 + y2*y2
                a3, b3, c3 = x3, y3, x3*x3 + y3*y3
                a4, b4, c4 = x4, y4, x4*x4 + y4*y4
                d = (a1*(b2*(c3 - c4) - c2*(b3 - b4) + (b3*c4 - c3*b4))
                     - b1*(a2*(c3 - c4) - c2*(a3 - a4) + (a3*c4 - c3*a4))
                     + c1*(a2*(b3 - b4) - b2*(a3 - a4) + (a3*b4 - b3*a4))
                     - (a2*(b3*c4 - c3*b4) - b2*(a3*c4 - c3*a4) + c2*(a3*b4 - b3*a4)))
                if d == 0:
                    return False
    return True

def _generate_candidates(max_coord, time_limit):
    """Generate candidate points sorted by distance from origin."""
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    return candidates

def _search_greedy(D, seed_pts, max_coord, time_limit):
    """Greedy search: try to add points in order of distance from origin."""
    current = list(seed_pts)
    best = list(current)
    
    candidates = _generate_candidates(max_coord, time_limit)
    
    for candidate in candidates:
        if time.time() > time_limit:
            break
        if candidate in current:
            continue
        if _try_add_point(current, D, candidate):
            current.append(candidate)
            if len(current) > len(best):
                best = list(current)
                if len(best) >= 12:
                    break
    return best

def _search_with_swaps(D, seed_pts, max_coord, time_limit):
    """Search with local swaps to escape local optima."""
    current = list(seed_pts)
    best = list(current)
    
    candidates = _generate_candidates(max_coord, time_limit)
    
    # First greedy pass
    for candidate in candidates:
        if time.time() > time_limit:
            break
        if candidate in current:
            continue
        if _try_add_point(current, D, candidate):
            current.append(candidate)
            if len(current) > len(best):
                best = list(current)
    
    # Try swaps: remove one point, add another
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for i in range(len(current)):
            if time.time() > time_limit:
                break
            saved = current.pop(i)
            for candidate in candidates:
                if time.time() > time_limit:
                    break
                if candidate in current or candidate == saved:
                    continue
                if _try_add_point(current, D, candidate):
                    current.append(candidate)
                    # Now try to add more
                    for extra in candidates:
                        if time.time() > time_limit:
                            break
                        if extra in current:
                            continue
                        if _try_add_point(current, D, extra):
                            current.append(extra)
                            if len(current) > len(best):
                                best = list(current)
                                improved = True
                    if len(current) > len(best):
                        best = list(current)
                    break
            current.insert(i, saved)
    
    return best

def construct():
    D = 1
    
    # Known valid 5-point set (Pythagorean)
    seed5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate(seed5, D):
        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in seed5]}
    
    best_pts = list(seed5)
    time_limit = _DEADLINE - 5.0
    
    # Try greedy search from seed5
    grown = _search_greedy(D, seed5, 150, time_limit)
    if len(grown) > len(best_pts):
        best_pts = grown
    
    # Try other seeds
    other_seeds = [
        [(0, 0), (3, 4), (0, 5)],
        [(0, 0), (5, 0), (0, 12)],
        [(0, 0), (9, 12), (0, 15)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (2, 0), (0, 3)],
        [(0, 0), (7, 24), (0, 25)],
        [(0, 0), (12, 35), (0, 37)],
        [(0, 0), (20, 21), (0, 29)],
        [(0, 0), (8, 15), (0, 17)],
        [(0, 0), (11, 60), (0, 61)],
        [(0, 0), (28, 45), (0, 53)],
        [(0, 0), (33, 56), (0, 65)],
        [(0, 0), (48, 55), (0, 73)],
        [(0, 0), (13, 84), (0, 85)],
        [(0, 0), (16, 63), (0, 65)],
        [(0, 0), (36, 77), (0, 85)],
    ]
    
    for seed in other_seeds:
        if time.time() > time_limit:
            break
        if _validate(seed, D):
            grown = _search_greedy(D, seed, 100, time_limit)
            if len(grown) > len(best_pts):
                best_pts = grown
    
    # Try swap-based search from best
    if time.time() < time_limit:
        grown = _search_with_swaps(D, best_pts, 120, time_limit)
        if len(grown) > len(best_pts):
            best_pts = grown
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
