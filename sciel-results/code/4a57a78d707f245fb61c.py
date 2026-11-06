import math
import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Leave 15s margin for the 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > _TIME_LIMIT

def _sq(n):
    return n * n

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return _sq(dx) + _sq(dy)

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) == 0

def _are_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    a1 = x1 - x2
    b1 = y1 - y2
    c1 = x2*x2 + y2*y2 - x1*x1 - y1*y1
    
    a2 = x1 - x3
    b2 = y1 - y3
    c2 = x3*x3 + y3*y3 - x1*x1 - y1*y1
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        return False
    
    D_num = c1 * b2 - c2 * b1
    E_num = a1 * c2 - a2 * c1
    
    return (det * (x4*x4 + y4*y4) + D_num * x4 + E_num * y4 
            - det * (x1*x1 + y1*y1) - D_num * x1 - E_num * y1) == 0

def _valid_set(pts):
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    for i in range(n):
        for j in range(i+1, n):
            dsq = _dist_sq(pts[i], pts[j])
            if not _is_square(dsq):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _format_output(pts):
    return {'D': 1, 'points': [[[p[0], 1], [p[1], 1]] for p in pts]}

def _pythagorean_triples(max_r):
    """Generate all (x, y) with x^2 + y^2 = r^2 for r in [1, max_r]."""
    pts = set()
    for r in range(1, max_r + 1):
        r2 = r * r
        for x in range(0, r + 1):
            y2 = r2 - x * x
            if y2 < 0:
                continue
            y = math.isqrt(y2)
            if y * y == y2:
                pts.add((x, y))
                pts.add((-x, y))
                pts.add((x, -y))
                pts.add((-x, -y))
                if x != 0 and x != y:
                    pts.add((y, x))
                    pts.add((-y, x))
                    pts.add((y, -x))
                    pts.add((-y, -x))
    return list(pts)

def _find_6th_point_parametric(base5, max_mn=200):
    """Find a 6th point using parametric Pythagorean triples.
    
    Key insight: if the 6th point P is at integer distance from origin (after translation),
    then P = k*(m²-n², 2mn) for some integers m,n,k. We search over (m,n,k) and check
    if P is at integer distance from all other base points.
    """
    ox, oy = base5[0]
    translated = [(x - ox, y - oy) for x, y in base5]
    
    # The first point is at origin after translation
    # Find all (m,n) such that the point (m²-n², 2mn) or its variants
    # are at integer distance from all translated base points
    
    found = []
    
    for m in range(1, max_mn + 1):
        if _deadline_exceeded():
            break
        for n in range(0, m):
            if _deadline_exceeded():
                break
            m2 = m * m
            n2 = n * n
            base_x = m2 - n2
            base_y = 2 * m * n
            
            # Try all sign combinations and swaps with various k values
            candidates = set()
            for sx in (1, -1):
                for sy in (1, -1):
                    candidates.add((sx * base_x, sy * base_y))
                    if base_x != base_y:
                        candidates.add((sx * base_y, sy * base_x))
            
            for cx, cy in candidates:
                if (cx, cy) in translated:
                    continue
                # Check if at integer distance from all translated base points
                ok = True
                for bx, by in translated:
                    dsq = (cx - bx)**2 + (cy - by)**2
                    if not _is_square(dsq):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Scale by k to get actual candidate
                # Actually, (cx, cy) itself is a candidate (k=1)
                # But we can also try multiples... however that changes distances
                # Let's just use the unscaled version first
                
                px, py = cx + ox, cy + oy
                found.append((px, py))
    
    if _deadline_exceeded():
        return None
    
    # Try each candidate
    for cand in found:
        if _deadline_exceeded():
            return None
        test_set = base5 + [cand]
        if _valid_set(test_set):
            return test_set
    
    return None

def _find_6th_point_grid(base5, max_r):
    """Find a 6th point at integer distance from all 5 base points using grid search."""
    ox, oy = base5[0]
    translated = [(x - ox, y - oy) for x, y in base5]
    
    candidates = _pythagorean_triples(max_r)
    
    valid_candidates = []
    for cx, cy in candidates:
        if (cx, cy) in translated:
            continue
        ok = True
        for bx, by in translated:
            dsq = (cx - bx)**2 + (cy - by)**2
            if not _is_square(dsq):
                ok = False
                break
        if not ok:
            continue
        px, py = cx + ox, cy + oy
        valid_candidates.append((px, py))
    
    if _deadline_exceeded():
        return None
    
    for cand in valid_candidates:
        if _deadline_exceeded():
            return None
        test_set = base5 + [cand]
        if _valid_set(test_set):
            return test_set
    
    return None

def _find_7th_point(base6, max_r):
    """Find a 7th point at integer distance from all 6 base points."""
    ox, oy = base6[0]
    translated = [(x - ox, y - oy) for x, y in base6]
    
    candidates = _pythagorean_triples(max_r)
    
    valid_candidates = []
    for cx, cy in candidates:
        if (cx, cy) in translated:
            continue
        ok = True
        for bx, by in translated:
            dsq = (cx - bx)**2 + (cy - by)**2
            if not _is_square(dsq):
                ok = False
                break
        if not ok:
            continue
        px, py = cx + ox, cy + oy
        valid_candidates.append((px, py))
    
    if _deadline_exceeded():
        return None
    
    for cand in valid_candidates:
        if _deadline_exceeded():
            return None
        test_set = base6 + [cand]
        if _valid_set(test_set):
            return test_set
    
    return None

def _try_multiple_seeds():
    """Try multiple known 5-point seeds to find a 6th point."""
    seeds = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        # Try other known configurations
        [(0, 0), (3, 4), (-3, 4), (0, 5), (5, 0)],  # May not be valid
        [(0, 0), (5, 0), (0, 5), (-5, 0), (0, -5)],  # Cross pattern
    ]
    
    for seed in seeds:
        if _deadline_exceeded():
            break
        if not _valid_set(seed):
            continue
        
        # Try parametric search
        result = _find_6th_point_parametric(seed, max_mn=150)
        if result is not None:
            return result
        
        # Try grid search
        for max_r in [100, 150, 200]:
            if _deadline_exceeded():
                break
            result = _find_6th_point_grid(seed, max_r)
            if result is not None:
                return result
    
    return None

def construct():
    # Known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify the base set
    if not _valid_set(best_points):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    # Strategy 1: Parametric search using Pythagorean triple parameterization
    if not _deadline_exceeded():
        result = _find_6th_point_parametric(best_points, max_mn=300)
        if result is not None:
            best_points = result
    
    # Strategy 2: Grid search with Pythagorean triple candidates
    if len(best_points) == 5 and not _deadline_exceeded():
        for max_r in [100, 150, 200, 300, 400, 500]:
            if _deadline_exceeded():
                break
            result = _find_6th_point_grid(best_points, max_r)
            if result is not None:
                best_points = result
                break
    
    # Strategy 3: Try multiple seeds
    if len(best_points) == 5 and not _deadline_exceeded():
        result = _try_multiple_seeds()
        if result is not None:
            best_points = result
    
    # Strategy 4: Brute force grid in larger region
    if len(best_points) == 5 and not _deadline_exceeded():
        ox, oy = best_points[0]
        max_coord = 200
        found = False
        
        for x in range(ox - max_coord, ox + max_coord + 1):
            if _deadline_exceeded() or found:
                break
            for y in range(oy - max_coord, oy + max_coord + 1):
                if _deadline_exceeded() or found:
                    break
                p = (x, y)
                if p in best_points:
                    continue
                ok = True
                for bp in best_points:
                    dsq = _dist_sq(p, bp)
                    if not _is_square(dsq):
                        ok = False
                        break
                if not ok:
                    continue
                test_set = best_points + [p]
                if _valid_set(test_set):
                    best_points = test_set
                    found = True
                    break
    
    # Try to find 7 points
    if len(best_points) == 6 and not _deadline_exceeded():
        for max_r in [100, 150, 200, 300]:
            if _deadline_exceeded():
                break
            result = _find_7th_point(best_points, max_r)
            if result is not None:
                best_points = result
                break
    
    return _format_output(best_points)
