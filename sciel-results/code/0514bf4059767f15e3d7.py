"""Integral point sets in general position - search for maximum size."""
import time
import random
from math import isqrt

_START = time.time()
_DEADLINE = _START + 145.0  # 145s safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
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
    # Precompute x^2 + y^2
    sqs = [(x*x + y*y) for x, y in pts]
    
    for i in range(n):
        x1, y1 = pts[i]
        c1 = sqs[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            c2 = sqs[j]
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                c3 = sqs[k]
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    x4, y4 = pts[l]
                    c4 = sqs[l]
                    # Determinant of 4x4 matrix
                    # | x1 y1 c1 1 |
                    # | x2 y2 c2 1 |
                    # | x3 y3 c3 1 |
                    # | x4 y4 c4 1 |
                    # Subtract row 4 from rows 1,2,3
                    # | x1-x4 y1-y4 c1-c4 0 |
                    # | x2-x4 y2-y4 c2-c4 0 |
                    # | x3-x4 y3-y4 c3-c4 0 |
                    # | x4   y4   c4     1 |
                    # Det = det of top-left 3x3
                    a1, b1, c1d = x1-x4, y1-y4, c1-c4
                    a2, b2, c2d = x2-x4, y2-y4, c2-c4
                    a3, b3, c3d = x3-x4, y3-y4, c3-c4
                    
                    det = a1 * (b2 * c3d - c2d * b3) - b1 * (a2 * c3d - c2d * a3) + c1d * (a2 * b3 - b2 * a3)
                    if det == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
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
    # Check distances from candidate to existing points
    for p in pts:
        dx = candidate[0] - p[0]
        dy = candidate[1] - p[1]
        d2 = dx*dx + D*dy*dy
        if not _is_square(d2):
            return False
    
    # Check collinearity: candidate with any pair in pts
    n = len(pts)
    x4, y4 = candidate
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx * (y4 - y1) - dy * (x4 - x1) == 0:
                return False
            
    # Check concyclicity: candidate with any triple in pts
    c4 = x4*x4 + y4*y4
    for i in range(n):
        x1, y1 = pts[i]
        c1 = x1*x1 + y1*y1
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            c2 = x2*x2 + y2*y2
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                c3 = x3*x3 + y3*y3
                
                a1, b1, c1d = x1-x4, y1-y4, c1-c4
                a2, b2, c2d = x2-x4, y2-y4, c2-c4
                a3, b3, c3d = x3-x4, y3-y4, c3-c4
                
                det = a1 * (b2 * c3d - c2d * b3) - b1 * (a2 * c3d - c2d * a3) + c1d * (a2 * b3 - b2 * a3)
                if det == 0:
                    return False
    return True

def _search_greedy(D, seed_pts, max_coord, time_limit, rng):
    """Greedy search: try to add points in order of distance from centroid."""
    current = list(seed_pts)
    best = list(current)
    
    # Generate candidates
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in current:
                candidates.append((x, y))
                
    # Sort by distance to centroid
    if current:
        cx = sum(p[0] for p in current) / len(current)
        cy = sum(p[1] for p in current) / len(current)
        candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    else:
        candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
        
    # Shuffle slightly to break ties or allow different paths if random restart is used
    # For pure greedy, we keep the sort. For random restart, we shuffle.
    if rng is not None:
        rng.shuffle(candidates)
        
    for candidate in candidates:
        if time.time() > time_limit:
            break
        if _try_add_point(current, D, candidate):
            current.append(candidate)
            if len(current) > len(best):
                best = list(current)
                if len(best) >= 12:
                    break
    return best

def construct():
    D = 1
    
    # Known valid 5-point set
    seed5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate(seed5, D):
        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in seed5]}
    
    best_pts = list(seed5)
    time_limit = _DEADLINE - 5.0
    
    # Try greedy search from seed5 with increasing ranges
    for max_coord in [50, 100, 150, 200]:
        if time.time() > time_limit:
            break
        grown = _search_greedy(D, seed5, max_coord, time_limit, rng=None)
        if len(grown) > len(best_pts):
            best_pts = grown
            if len(best_pts) >= 12:
                break
        
    # Try random restarts from seed5
    rng = random.Random(42)
    for restart in range(10):
        if time.time() > time_limit:
            break
        # Use a smaller range for random restarts to be faster
        grown = _search_greedy(D, seed5, 100, time_limit, rng=rng)
        if len(grown) > len(best_pts):
            best_pts = grown
            if len(best_pts) >= 12:
                break

    # Try other seeds if time permits
    other_seeds = [
        [(0, 0), (3, 4), (0, 5)],
        [(0, 0), (5, 0), (0, 12)],
        [(0, 0), (9, 12), (0, 15)],
    ]
    
    for seed in other_seeds:
        if time.time() > time_limit:
            break
        if _validate(seed, D):
            grown = _search_greedy(D, seed, 100, time_limit, rng=None)
            if len(grown) > len(best_pts):
                best_pts = grown
            # Random restarts for this seed
            for r in range(5):
                if time.time() > time_limit:
                    break
                grown = _search_greedy(D, seed, 80, time_limit, rng=rng)
                if len(grown) > len(best_pts):
                    best_pts = grown

    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
