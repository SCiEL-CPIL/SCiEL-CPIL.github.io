"""Integral point sets in general position - search for maximum size."""
import time
import random
from fractions import Fraction
from math import gcd, isqrt

# Deadline guard
_DEADLINE = time.time() + 130  # 130s wall-clock, 20s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    return gcd(abs(a), abs(b))

def _is_valid(points, D):
    """Check all constraints: distinct, no three collinear, no four concyclic, all distances rational."""
    n = len(points)
    if n < 3:
        return False
    # Check distinctness and rational distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + D * dy * dy
            if d2 <= 0:
                return False
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Cross product
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                if dx1 * dy2 - dy1 * dx2 == 0:
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points concyclic iff the determinant of the circle equation is zero
                    # Circle: x^2 + y^2 + Dx + Ey + F = 0
                    # For point (x,y): x^2+y^2+Dx+Ey+F=0
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # Build 4x4 system and check determinant
                    # [x1^2+y1^2, x1, y1, 1]
                    # [x2^2+y2^2, x2, y2, 1]
                    # [x3^2+y3^2, x3, y3, 1]
                    # [x4^2+y4^2, x4, y4, 1]
                    # det = 0 means concyclic
                    a1 = x1*x1 + D*y1*y1
                    a2 = x2*x2 + D*y2*y2
                    a3 = x3*x3 + D*y3*y3
                    a4 = x4*x4 + D*y4*y4
                    # Compute determinant of:
                    # |a1 x1 y1 1|
                    # |a2 x2 y2 1|
                    # |a3 x3 y3 1|
                    # |a4 x4 y4 1|
                    # Subtract row 1 from rows 2,3,4 to simplify
                    # |a1 x1 y1 1|
                    # |a2-a1 x2-x1 y2-y1 0|
                    # |a3-a1 x3-x1 y3-y1 0|
                    # |a4-a1 x4-x1 y4-y1 0|
                    # det = a1 * det([[x2-x1,y2-y1],[x3-x1,y3-y1],[x4-x1,y4-y1]] 3x3 without first col)
                    # Actually: expand along last column:
                    # det = 1 * det([[a2-a1, x2-x1, y2-y1],[a3-a1, x3-x1, y3-y1],[a4-a1, x4-x1, y4-y1]])
                    b2 = a2 - a1
                    b3 = a3 - a1
                    b4 = a4 - a1
                    c2 = x2 - x1
                    c3 = x3 - x1
                    c4 = x4 - x1
                    d2 = y2 - y1
                    d3 = y3 - y1
                    d4 = y4 - y1
                    # 3x3 det:
                    # |b2 c2 d2|
                    # |b3 c3 d3|
                    # |b4 c4 d4|
                    det3 = (b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4))
                    if det3 == 0:
                        return False
    return True

def _try_add(points, x, y, D):
    """Try to add point (x,y) to the set."""
    new_points = points + [(x, y)]
    return _is_valid(new_points, D)

def _search_incremental(start_points, max_points, time_budget, D):
    """Incrementally try to add points to the set."""
    current = list(start_points)
    n = len(current)
    # Track which points we've tried
    tried = set()
    
    while n < max_points and _time_left() > 5:
        progress = False
        # Try random candidate points
        for attempt in range(50000):
            if _time_left() < 5:
                break
            # Generate candidate: try points near existing ones or in a grid
            # Strategy: try integer points in a reasonable range
            # Scale based on existing point spread
            xs = [p[0] for p in current]
            ys = [p[1] for p in current]
            x_min = min(xs) - 50
            x_max = max(xs) + 50
            y_min = min(ys) - 50
            y_max = max(ys) + 50
            
            # Random candidate
            cx = random.randint(x_min, x_max)
            cy = random.randint(y_min, y_max)
            
            if (cx, cy) in tried:
                continue
            tried.add((cx, cy))
            
            # Quick check: distance to all existing points must be perfect square
            ok = True
            for px, py in current:
                dx = cx - px
                dy = cy - py
                d2 = dx*dx + D*dy*dy
                if not _is_square(d2):
                    ok = False
                    break
            
            if not ok:
                continue
            
            # Full validity check
            if _try_add(current, cx, cy, D):
                current.append((cx, cy))
                n += 1
                progress = True
                tried = set()  # Reset tried set
                break
        
        if not progress:
            # Try a different strategy: systematic search
            # Try points that form right triangles with existing points
            found = False
            for px, py in current:
                for r in range(1, 200):
                    if _time_left() < 5:
                        break
                    # Points at distance r from (px,py) with integer coordinates
                    # x^2 + D*y^2 = r^2
                    for dx in range(-r, r+1):
                        dy2 = r*r - dx*dx
                        if dy2 < 0:
                            continue
                        # Need D*dy^2 = dy2, so dy2 must be divisible by D and dy2/D must be a perfect square
                        if D == 0:
                            continue
                        if dy2 % D != 0:
                            continue
                        dy_sq = dy2 // D
                        dy = isqrt(dy_sq)
                        if dy*dy != dy_sq:
                            continue
                        for sy in [dy, -dy]:
                            cx = px + dx
                            cy = py + sy
                            if (cx, cy) in tried:
                                continue
                            tried.add((cx, cy))
                            if _try_add(current, cx, cy, D):
                                current.append((cx, cy))
                                n += 1
                                found = True
                                tried = set()
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
    
    return current

def _search_from_known(known_sets, max_points, time_budget, D):
    """Try to extend known sets."""
    best = None
    for start in known_sets:
        if _time_left() < 10:
            break
        result = _search_incremental(start, max_points, time_budget, D)
        if best is None or len(result) > len(best):
            best = result
    return best

def _generate_candidates_D(D, limit=1000):
    """Generate candidate points (x,y) such that x^2 + D*y^2 is a perfect square."""
    candidates = set()
    for y in range(-50, 51):
        for x in range(-100, 101):
            d2 = x*x + D*y*y
            if d2 >= 0 and _is_square(d2):
                candidates.add((x, y))
    return list(candidates)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 130
    
    # Known integral point sets (integer points with integer distances)
    # Start from various known configurations
    known_sets_D1 = [
        # 5-point set from seed
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        # Other known configurations
        [(0, 0), (5, 0), (0, 12)],
        [(0, 0), (3, 4), (5, 0)],
        [(0, 0), (7, 24), (15, 8)],
        [(0, 0), (9, 12), (15, 0)],
        [(0, 0), (12, 5), (13, 0)],
        [(0, 0), (20, 21), (29, 0)],
        [(0, 0), (8, 15), (17, 0)],
        [(0, 0), (11, 60), (61, 0)],
        [(0, 0), (28, 45), (53, 0)],
        [(0, 0), (33, 56), (65, 0)],
        [(0, 0), (16, 63), (65, 0)],
        [(0, 0), (48, 55), (73, 0)],
        [(0, 0), (65, 72), (97, 0)],
        [(0, 0), (20, 99), (101, 0)],
        [(0, 0), (99, 20), (101, 0)],
    ]
    
    # First, verify which known sets are valid
    valid_starts_D1 = []
    for s in known_sets_D1:
        if len(s) >= 3 and _is_valid(s, 1):
            valid_starts_D1.append(s)
    
    # Try to extend each valid set in D=1
    best_result_D1 = None
    for start in valid_starts_D1:
        if _time_left() < 15:
            break
        result = _search_incremental(start, 12, 130, 1)
        if best_result_D1 is None or len(result) > len(best_result_D1):
            best_result_D1 = result
    
    # Now search in D=2, 3, 5, 6, 7
    best_result = best_result_D1
    best_D = 1
    
    for D in [2, 3, 5, 6, 7]:
        if _time_left() < 15:
            break
        
        # Generate candidate points for this D
        candidates = _generate_candidates_D(D)
        
        # Try to build a set incrementally
        best_for_D = None
        
        # Start with (0,0) and try to add points
        base = [(0, 0)]
        for cx, cy in candidates:
            if (cx, cy) == (0, 0):
                continue
            if _try_add(base, cx, cy, D):
                base.append((cx, cy))
                if len(base) >= 3:
                    break
        
        if len(base) >= 3:
            result = _search_incremental(base, 12, 130, D)
            if best_for_D is None or len(result) > len(best_for_D):
                best_for_D = result
        
        # Try other starting points
        for start_pt in candidates[:20]:
            if _time_left() < 15:
                break
            base2 = [start_pt]
            for cx, cy in candidates:
                if (cx, cy) == start_pt:
                    continue
                if _try_add(base2, cx, cy, D):
                    base2.append((cx, cy))
                    if len(base2) >= 3:
                        break
            
            if len(base2) >= 3:
                result = _search_incremental(base2, 12, 130, D)
                if best_for_D is None or len(result) > len(best_for_D):
                    best_for_D = result
        
        if best_for_D is not None:
            if best_result is None or len(best_for_D) > len(best_result):
                best_result = best_for_D
                best_D = D
    
    # If we haven't found anything better, try a fresh random search
    if best_result is None or len(best_result) <= 5:
        # Try random initialization
        for trial in range(50):
            if _time_left() < 15:
                break
            # Start with a random triangle with integer distances
            # Pick three random points and check
            base = [(0, 0)]
            for _ in range(2000):
                if _time_left() < 15:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) == (0, 0):
                    continue
                if not _is_square(x*x + y*y):
                    continue
                # Check collinearity with existing
                collinear = False
                for px, py in base:
                    if len(base) >= 2:
                        for qx, qy in base:
                            if (x-px)*(qy-py) == (y-py)*(qx-px):
                                collinear = True
                                break
                    if collinear:
                        break
                if not collinear:
                    base.append((x, y))
                    break
            
            if len(base) >= 3:
                result = _search_incremental(base, 12, 130, 1)
                if best_result is None or len(result) > len(best_result):
                    best_result = result
    
    # Fallback: return the 5-point set
    if best_result is None or len(best_result) < 3:
        best_result = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_D = 1
    
    # Final validation
    if not _is_valid(best_result, best_D):
        best_result = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_D = 1
    
    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_result]}
