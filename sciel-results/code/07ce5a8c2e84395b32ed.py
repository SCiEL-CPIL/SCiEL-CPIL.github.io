"""Search for integral point sets with rational distances, no 3 collinear, no 4 concyclic."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_pairwise_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + D*dy*dy
            num, den = d2.numerator, d2.denominator
            if num < 0 or den < 0:
                return False
            inner = num * den
            if inner < 0:
                return False
            if not _is_square(inner):
                return False
    return True

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                cross = dx1*dy2 - dy1*dx2
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for m in range(k+1, n):
                    rows = []
                    for idx in [i, j, k, m]:
                        x = points[idx][0]
                        y = points[idx][1]
                        rows.append([x*x + D*y*y, x, y, Fraction(1)])
                    a = rows[0]; b = rows[1]; c = rows[2]; d = rows[3]
                    det = (a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[1]*d[3]-c[3]*d[1]) + b[3]*(c[1]*d[2]-c[2]*d[1]))
                         - a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[2]-c[2]*d[0]))
                         + a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1]) - b[1]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[1]-c[1]*d[0]))
                         - a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1]) - b[1]*(c[0]*d[2]-c[2]*d[0]) + b[2]*(c[0]*d[1]-c[1]*d[0])))
                    if det == 0:
                        return False
    return True

def _valid_set(points, D):
    if not _check_pairwise_rational(points, D):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _find_6th_point(base, deadline):
    """Find a 6th integer point with integer distances to all base points."""
    # For each pair of base points, compute circle intersections
    # A point P at distance r1 from A and r2 from B satisfies:
    # |P-A|^2 = r1^2, |P-B|^2 = r2^2
    # This gives a linear equation in P, so P lies on a line (radical axis)
    # Then we can search for integer points on that line at integer distances
    
    n = len(base)
    # Try all pairs of base points
    for i in range(n):
        for j in range(i+1, n):
            ax, ay = base[i]
            bx, by = base[j]
            ab2 = (ax-bx)**2 + (ay-by)**2
            ab = math.isqrt(ab2)
            if ab*ab != ab2:
                continue
            
            # Try various distance pairs (r1, r2)
            for r1 in range(1, 300):
                if time.time() > deadline:
                    return None
                for r2 in range(1, 300):
                    # Check triangle inequality with AB
                    if r1 + r2 <= ab or abs(r1-r2) >= ab:
                        continue
                    # Compute intersection of circles
                    # x^2 + y^2 = r1^2 (relative to A)
                    # (x-(bx-ax))^2 + (y-(by-ay))^2 = r2^2
                    # Linear: 2*(bx-ax)*x + 2*(by-ay)*y = r1^2 - r2^2 + (bx-ax)^2 + (by-ay)^2
                    dx = bx - ax
                    dy = by - ay
                    rhs = r1*r1 - r2*r2 + dx*dx + dy*dy
                    # 2*dx*x + 2*dy*y = rhs
                    # x = (rhs - 2*dy*y) / (2*dx) if dx != 0
                    # We need integer solutions
                    # Try: if 2*dx divides (rhs - 2*dy*y)
                    # Or parametrize: find integer (x,y) on line 2*dx*x + 2*dy*y = rhs
                    # and at distance r1 from (ax,ay)
                    
                    # For each possible y, check if x is integer
                    # 2*dx*x = rhs - 2*dy*y
                    # x = (rhs - 2*dy*y) / (2*dx)
                    # Need 2*dx | (rhs - 2*dy*y)
                    # i.e., rhs - 2*dy*y ≡ 0 (mod 2*dx)
                    # 2*dy*y ≡ rhs (mod 2*dx)
                    
                    # This is a linear congruence. Let's just try y in a range
                    # and check if x is integer and |P-A| = r1
                    
                    # Actually, let's be smarter: the intersection of two circles
                    # is at most 2 points. Let's compute them directly.
                    # Midpoint of AB: M = ((ax+bx)/2, (ay+by)/2)
                    # Distance from M to each intersection point:
                    # h = sqrt(r1^2 - (ab/2)^2) if r1^2 >= (ab/2)^2
                    # But we need integer coordinates...
                    
                    # Let's just brute force y in a bounded range
                    # The intersection points are bounded
                    max_y = max(ay, by) + max(r1, r2) + 10
                    min_y = min(ay, by) - max(r1, r2) - 10
                    max_x = max(ax, bx) + max(r1, r2) + 10
                    min_x = min(ax, bx) - max(r1, r2) - 10
                    
                    # Try integer points in the bounding box
                    for py in range(min_y, max_y+1):
                        if time.time() > deadline:
                            return None
                        for px in range(min_x, max_x+1):
                            if time.time() > deadline:
                                return None
                            # Check distance to A is r1
                            d1 = (px-ax)**2 + (py-ay)**2
                            if d1 != r1*r1:
                                continue
                            # Check distance to B is r2
                            d2 = (px-bx)**2 + (py-by)**2
                            if d2 != r2*r2:
                                continue
                            # Found candidate! Check all other base points
                            cand = (px, py)
                            if cand in base:
                                continue
                            ok = True
                            for k in range(n):
                                if k == i or k == j:
                                    continue
                                dxk = px - base[k][0]
                                dyk = py - base[k][1]
                                dk2 = dxk*dxk + dyk*dyk
                                sk = math.isqrt(dk2)
                                if sk*sk != dk2:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            # Check no three collinear
                            new_pts = base + [cand]
                            if not _no_three_collinear(new_pts):
                                continue
                            # Check no four concyclic
                            if not _no_four_concyclic(new_pts, 1):
                                continue
                            return new_pts
    return None

def _search_6_points(deadline):
    """Search for 6-point integral point set."""
    # Known 5-point set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to extend the known 5-point set
    result = _find_6th_point(base5, deadline)
    if result:
        return result
    
    # Try other 5-point bases
    # Generate candidate 5-point sets and try to extend
    # Use the structure: find 4 points first, then add 5th, then 6th
    
    # Try a different approach: search for 6 points directly
    # Start with a small set and grow
    
    # Try base sets with different structures
    base_sets = [
        [(0, 0), (3, 4), (5, 12), (15, 8), (8, 15)],
        [(0, 0), (5, 12), (9, 40), (28, 45), (16, 63)],
        [(0, 0), (3, 4), (7, 24), (15, 8), (8, 15)],
    ]
    
    for base in base_sets:
        if time.time() > deadline:
            break
        # Verify base is valid
        if not _valid_set(base, 1):
            continue
        result = _find_6th_point(base, deadline)
        if result:
            return result
    
    return None

def construct():
    start = time.time()
    deadline = start + 130  # safety margin
    
    # Try to find 6 points
    result = _search_6_points(deadline)
    
    if result:
        pts = result
        D = 1
    else:
        # Fallback to known 5-point set
        pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        D = 1
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
