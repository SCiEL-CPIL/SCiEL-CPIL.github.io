"""Improved program: Efficient search for integral point sets."""
import time
from fractions import Fraction
from itertools import combinations
from math import isqrt

_DEADLINE = 140.0  # safety margin before 150s kill

def _is_square(n):
    """Check if non-negative integer n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _rational_dist2(p, q):
    """Squared distance between two points as a Fraction."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _check_rational_dist(p, q):
    """Check if distance between p and q is rational."""
    d2 = _rational_dist2(p, q)
    num, den = d2.numerator, d2.denominator
    prod = num * den
    return _is_square(prod)

def _collinear(p, q, r):
    """Check if three points are collinear using cross product."""
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - p[0]
    dy2 = r[1] - p[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def _concyclic(p, q, r, s):
    """Check if four points are concyclic using determinant of circle equation."""
    def f(pt):
        return (pt[0]*pt[0] + pt[1]*pt[1], pt[0], pt[1], 1)
    A, B, C, D = f(p), f(q), f(r), f(s)
    # 4x4 determinant
    det = (A[0]*(B[1]*(C[2]*D[3]-C[3]*D[2]) - B[2]*(C[1]*D[3]-C[3]*D[1]) + B[3]*(C[1]*D[2]-C[2]*D[1]))
          - A[1]*(B[0]*(C[2]*D[3]-C[3]*D[2]) - B[2]*(C[0]*D[3]-C[3]*D[0]) + B[3]*(C[0]*D[2]-C[2]*D[0]))
          + A[2]*(B[0]*(C[1]*D[3]-C[3]*D[1]) - B[1]*(C[0]*D[3]-C[3]*D[0]) + B[3]*(C[0]*D[1]-C[1]*D[0]))
          - A[3]*(B[0]*(C[1]*D[2]-C[2]*D[1]) - B[1]*(C[0]*D[2]-C[2]*D[0]) + B[2]*(C[0]*D[1]-C[1]*D[0])))
    return det == 0

def _check_valid(points):
    """Full validity check: all pairwise distances rational, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
            if not _check_rational_dist(points[i], points[j]):
                return False
    for combo in combinations(range(n), 3):
        if _collinear(points[combo[0]], points[combo[1]], points[combo[2]]):
            return False
    for combo in combinations(range(n), 4):
        if _concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
            return False
    return True

def _circle_intersections(p1, p2, r1, r2):
    """Find intersection points of two circles. Returns list of (Fraction, Fraction) or empty."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d2 = dx*dx + dy*dy
    if d2 == 0:
        return []
    
    num, den = d2.numerator, d2.denominator
    prod = num * den
    if not _is_square(prod):
        return []
    
    r_val = isqrt(prod)
    d_val = Fraction(r_val, den)
    
    # a = (r1^2 - r2^2 + d^2) / (2d)
    a = (r1*r1 - r2*r2 + d2) / (2 * d_val)
    h2 = r1*r1 - a*a
    if h2 < 0:
        return []
    
    h_num, h_den = h2.numerator, h2.denominator
    h_prod = h_num * h_den
    if not _is_square(h_prod):
        return []
    
    h_r = isqrt(h_prod)
    h_val = Fraction(h_r, h_den)
    
    # Midpoint offset
    mx = p1[0] + a * dx / d_val
    my = p1[1] + a * dy / d_val
    
    # Perpendicular direction
    px = -dy / d_val * h_val
    py = dx / d_val * h_val
    
    pt1 = (mx + px, my + py)
    pt2 = (mx - px, my - py)
    
    results = []
    for pt in [pt1, pt2]:
        x, y = pt
        if isinstance(x, Fraction) and isinstance(y, Fraction):
            results.append(pt)
    return results

def _search_incremental(starting_points, max_points=7, time_limit=135.0):
    """Incrementally add points using circle intersections."""
    start_time = time.time()
    best = list(starting_points)
    current = list(starting_points)
    
    for target_n in range(len(current)+1, max_points+1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        # Try increasing radii
        for max_radius in [50, 100, 200, 400]:
            if time.time() - start_time > time_limit:
                break
                
            candidates = set()
            n = len(current)
            
            # Use the first two points to generate candidates
            # This is much faster than all pairs
            p1, p2 = current[0], current[1]
            d12_sq = _rational_dist2(p1, p2)
            d12_num, d12_den = d12_sq.numerator, d12_sq.denominator
            d12_prod = d12_num * d12_den
            if not _is_square(d12_prod):
                continue
            d12_val = Fraction(isqrt(d12_prod), d12_den)
            
            for r1 in range(1, max_radius+1):
                if time.time() - start_time > time_limit:
                    break
                for r2 in range(1, max_radius+1):
                    # Quick check: triangle inequality
                    if r1 + r2 < d12_val or abs(r1 - r2) > d12_val:
                        continue
                    pts = _circle_intersections(p1, p2, Fraction(r1), Fraction(r2))
                    for pt in pts:
                        # Check if pt has rational distance to all other current points
                        ok = True
                        for k in range(2, n):
                            if not _check_rational_dist(pt, current[k]):
                                ok = False
                                break
                        if ok:
                            # Check collinearity
                            for i in range(n):
                                for j in range(i+1, n):
                                    if _collinear(pt, current[i], current[j]):
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if ok:
                                # Check concyclicity
                                if n >= 3:
                                    for combo in combinations(range(n), 3):
                                        if _concyclic(pt, current[combo[0]], current[combo[1]], current[combo[2]]):
                                            ok = False
                                            break
                                if ok:
                                    candidates.add(pt)
            
            # Try to find one candidate that extends the set
            for cand in candidates:
                if time.time() - start_time > time_limit:
                    break
                new_set = current + [cand]
                if _check_valid(new_set):
                    current = new_set
                    found = True
                    if len(current) > len(best):
                        best = list(current)
                    break
            
            if found:
                break
    
    return best

def construct():
    start_time = time.time()
    
    # Known valid 5-point set from previous runs
    base_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    
    best = base_points
    
    # Try to extend from the base set
    try:
        extended = _search_incremental(base_points, max_points=7, time_limit=130.0)
        if len(extended) > len(best):
            best = extended
    except Exception:
        pass
    
    # If we still only have 5, try a different starting configuration
    if len(best) <= 5 and time.time() - start_time < 100:
        # Try another known small set
        # This set is a 4-point set: (0,0), (1,0), (0,1), (1,1) is invalid.
        # Let's try: (0,0), (4,0), (0,3), (4,3) is invalid.
        # Try: (0,0), (5,0), (0,4), (3,4) -> d(5,0;3,4)=sqrt(4+16)=sqrt20 no.
        
        # Let's try a set derived from a 3-4-5 triangle
        # P1(0,0), P2(3,0), P3(0,4)
        # P4?
        # d(P4, P1), d(P4, P2), d(P4, P3) integers.
        # Try P4(3,4). d(0,0;3,4)=5. d(3,0;3,4)=4. d(0,4;3,4)=3.
        # So (0,0), (3,0), (0,4), (3,4) is a rectangle. Invalid.
        
        # We need a non-rectangular set.
        # How about P1(0,0), P2(4,0), P3(0,3).
        # P4(4,3) is rectangle.
        # Try P4 such that d(P4,P1)=5, d(P4,P2)=3, d(P4,P3)=4?
        # Circle(0,0,5), Circle(4,0,3).
        # Intersections:
        # x^2+y^2=25
        # (x-4)^2+y^2=9 => x^2-8x+16+y^2=9 => 25-8x+16=9 => 41-8x=9 => 8x=32 => x=4.
        # 16+y^2=25 => y=3.
        # So P4(4,3). Same rectangle.
        
        # Try different radii.
        # P1(0,0), P2(4,0), P3(0,3).
        # P4 with d(P4,P1)=5, d(P4,P2)=5?
        # x^2+y^2=25, (x-4)^2+y^2=25 => x=2.
        # 4+y^2=25 => y=sqrt(21). Not rational.
        
        # It is hard to find 4-point sets that are not concyclic.
        # Any 4 points with integer distances are likely to be concyclic if they are "nice".
        # Actually, no.
        # Example: (0,0), (1,0), (0,1), (1,1) is concyclic.
        # (0,0), (2,0), (0,1), (2,1) is concyclic.
        
        # Let's rely on the search from the 5-point set.
        
        # If search fails, we return the 5-point set.
        pass
    
    # Format output
    points_out = []
    for pt in best:
        x, y = pt
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': 1, 'points': points_out}
