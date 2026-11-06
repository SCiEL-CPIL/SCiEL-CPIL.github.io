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
                # Add all sign combinations and swaps
                pts.add((x, y))
                pts.add((-x, y))
                pts.add((x, -y))
                pts.add((-x, -y))
                if x != 0 and x != y:
                    pts.add((y, x))
                    pts.add((-y, x))
                    pts.add((y, -x))
                    pts.add((-y, -x))
                elif x == 0 and y != 0:
                    pass  # already covered
                elif x != 0 and x == y:
                    pass  # already covered
    return list(pts)

def _find_6th_point(base5, max_r):
    """Find a 6th point at integer distance from all 5 base points."""
    # Translate base so first point is at origin
    ox, oy = base5[0]
    translated = [(x - ox, y - oy) for x, y in base5]
    
    # Get all lattice points at integer distance from origin
    candidates = _pythagorean_triples(max_r)
    
    # Filter: must be at integer distance from all 5 translated base points
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
        # Translate back
        px, py = cx + ox, cy + oy
        valid_candidates.append((px, py))
    
    if _deadline_exceeded():
        return None
    
    # Try each candidate
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

def _try_alternative_bases(max_r):
    """Try different 5-point seeds to find a 6th point."""
    # Generate small integral point sets and try to extend
    # Start with points on circles around origin
    base_pts = [(0, 0)]
    
    # Find 4 more points that form a valid set with origin
    candidates = _pythagorean_triples(max_r // 2)
    
    # Try to build a 5-point set from origin + 4 others
    # Then try to find a 6th
    found_5s = []
    
    # Use the known 5-point set as primary
    known5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    found_5s.append(known5)
    
    # Try symmetric configurations
    # Equilateral-ish: use Pythagorean triples
    # (3,4,5), (5,12,13), (8,15,17), (7,24,25), (20,21,29)
    # Try combinations that give integer mutual distances
    
    # Point at (3,4): dist to origin = 5
    # Point at (5,12): dist to origin = 13, dist to (3,4) = sqrt(4+64)=sqrt(68) no
    # Point at (8,15): dist to origin = 17, dist to (3,4) = sqrt(25+121)=sqrt(146) no
    # Point at (7,24): dist to origin = 25, dist to (3,4) = sqrt(16+400)=sqrt(416) no
    # Point at (20,21): dist to origin = 29, dist to (3,4) = sqrt(289+289)=sqrt(578) no
    
    # Try (3,4), (-3,4): dist between = 6, dist to origin = 5 each
    # Point (0, 5): dist to (3,4) = sqrt(9+1)=sqrt(10) no
    # Point (0, -5): same issue
    # Point (6, 0): dist to (3,4) = sqrt(9+16)=5, dist to (-3,4) = sqrt(81+16)=sqrt(97) no
    # Point (4, 3): dist to origin = 5, dist to (3,4) = sqrt(1+1)=sqrt(2) no
    
    # Try (0,5), (5,0): dist = sqrt(50) no
    # Try (3,4), (4,3): dist = sqrt(1+1) no
    
    # The known 5-point set is likely the best seed. Let's focus on finding
    # a 6th point with a larger search radius.
    
    return found_5s

def construct():
    # Known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify the base set
    if not _valid_set(best_points):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    # Try to find 6 points using Pythagorean triple-based search
    # The 5 points span roughly x: -56..16, y: 0..63
    # A 6th point needs integer distance to all 5, so it must be on
    # intersection of 5 circles of integer radii centered at the 5 points
    
    # Start with moderate radius, increase if time permits
    for max_r in [80, 120, 160, 200, 300, 400, 500, 700, 1000]:
        if _deadline_exceeded():
            break
        result = _find_6th_point(best_points, max_r)
        if result is not None:
            best_points = result
            break
    
    # Try to find 7 points
    if len(best_points) == 6 and not _deadline_exceeded():
        for max_r in [80, 120, 160, 200, 300, 400]:
            if _deadline_exceeded():
                break
            result = _find_7th_point(best_points, max_r)
            if result is not None:
                best_points = result
                break
    
    # If we only have 5 points, try alternative approaches
    if len(best_points) == 5 and not _deadline_exceeded():
        # Try a brute-force grid search in a larger region
        # But use the circle-based prefilter
        ox, oy = best_points[0]
        max_coord = 150
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
    
    return _format_output(best_points)
