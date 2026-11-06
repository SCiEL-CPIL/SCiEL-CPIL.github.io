"""Rewritten program for integral point sets in general position."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_DEADLINE = 150.0
_START_TIME = None

def _now():
    return time.time() - _START_TIME

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sqfree(n):
    if n == 0:
        return 0
    res = 1
    p = 2
    temp = n
    while p * p <= temp:
        if temp % (p * p) == 0:
            while temp % (p * p) == 0:
                temp //= (p * p)
            res *= p
        elif temp % p == 0:
            temp //= p
        p += 1 if p == 2 else 2
    if temp > 1:
        res *= temp
    return res

def _dist_sq(p1, p2, D):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy * D

def _is_rational_dist_sq(v, D):
    if v < 0:
        return False
    num = v.numerator
    den = v.denominator
    return _is_square(num) and _is_square(den)

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _det4(m):
    return (
        m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
        - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
        + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
        - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
    )

def _check_no_four_concyclic(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    r1 = [p1[0]*p1[0] + D*p1[1]*p1[1], p1[0], D*p1[1], 1]
                    r2 = [p2[0]*p2[0] + D*p2[1]*p2[1], p2[0], D*p2[1], 1]
                    r3 = [p3[0]*p3[0] + D*p3[1]*p3[1], p3[0], D*p3[1], 1]
                    r4 = [p4[0]*p4[0] + D*p4[1]*p4[1], p4[0], D*p4[1], 1]
                    if _det4([r1, r2, r3, r4]) == 0:
                        return False
    return True

def _validate(points, D):
    set_pts = set(points)
    if len(set_pts) != len(points):
        return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d2 = _dist_sq(points[i], points[j], D)
            if not _is_rational_dist_sq(d2, D):
                return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def _pythagorean_triples(max_c):
    """Generate primitive and non-primitive Pythagorean triples with hypotenuse <= max_c."""
    triples = []
    for m in range(2, int(math.sqrt(max_c)) + 2):
        for n in range(1, m):
            if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                a = m*m - n*n
                b = 2*m*n
                c = m*m + n*n
                if c <= max_c:
                    triples.append((a, b, c))
                    triples.append((b, a, c))
    return triples

def _circle_intersection(p1, p2, r1, r2):
    """Find intersection points of two circles with centers p1, p2 and radii r1, r2.
    Returns list of (x, y) Fractions or empty list."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx*dx + dy*dy
    if d2 == 0:
        return []
    
    # Distance between centers
    d = math.sqrt(float(d2)) if d2 > 0 else 0
    # Check if circles intersect
    if r1 + r2 < d or abs(r1 - r2) > d:
        return []
    
    # a = distance from p1 to the line of intersection
    a = (r1*r1 - r2*r2 + d2) / (2 * d2)
    # h = half-distance between intersection points
    h2 = r1*r1 - a*a
    if h2 < 0:
        return []
    h = math.sqrt(float(h2))
    
    # Point on line p1->p2 at distance a from p1
    mx = x1 + a * dx
    my = y1 + a * dy
    
    # Perpendicular direction
    px = -dy / d
    py = dx / d
    
    # Two intersection points
    x_a = mx + h * px
    y_a = my + h * py
    x_b = mx - h * px
    y_b = my - h * py
    
    return [(Fraction(x_a).limit_denominator(10**20), Fraction(y_a).limit_denominator(10**20)),
            (Fraction(x_b).limit_denominator(10**20), Fraction(y_b).limit_denominator(10**20))]

def _rational_circle_intersection(p1, p2, r1_sq, r2_sq, D=1):
    """Find rational intersection points of two circles in the metric (x, y*sqrt(D)).
    Circles: (x-x1)^2 + D*(y-y1)^2 = r1_sq, (x-x2)^2 + D*(y-y2)^2 = r2_sq
    Returns list of (x, y) as Fractions."""
    x1, y1 = p1
    x2, y2 = p2
    
    # Subtract equations:
    # (x-x1)^2 - (x-x2)^2 + D*((y-y1)^2 - (y-y2)^2) = r1_sq - r2_sq
    # Expand:
    # x^2 - 2x*x1 + x1^2 - x^2 + 2x*x2 - x2^2 + D*(y^2 - 2y*y1 + y1^2 - y^2 + 2y*y2 - y2^2) = r1_sq - r2_sq
    # 2x*(x2-x1) + (x1^2 - x2^2) + D*2y*(y2-y1) + D*(y1^2 - y2^2) = r1_sq - r2_sq
    # 2*(x2-x1)*x + 2*D*(y2-y1)*y = r1_sq - r2_sq - x1^2 + x2^2 - D*y1^2 + D*y2^2
    
    dx = x2 - x1
    dy = y2 - y1
    rhs = r1_sq - r2_sq - x1*x1 + x2*x2 - D*y1*y1 + D*y2*y2
    
    # Linear equation: 2*dx*x + 2*D*dy*y = rhs
    # If dx != 0: x = (rhs - 2*D*dy*y) / (2*dx)
    # Substitute into first circle equation.
    
    # Case 1: dx != 0
    if dx != 0:
        # x = (rhs - 2*D*dy*y) / (2*dx)
        # Let A = rhs, B = 2*D*dy, C = 2*dx
        # x = (A - B*y) / C
        A = rhs
        B = 2 * D * dy
        C = 2 * dx
        
        # Substitute into (x-x1)^2 + D*(y-y1)^2 = r1_sq
        # ((A - B*y)/C - x1)^2 + D*(y-y1)^2 = r1_sq
        # ((A - B*y - C*x1)/C)^2 + D*(y-y1)^2 = r1_sq
        # (A - B*y - C*x1)^2 / C^2 + D*(y-y1)^2 = r1_sq
        # (A - B*y - C*x1)^2 + D*C^2*(y-y1)^2 = r1_sq * C^2
        
        # Let's expand:
        # (A - C*x1 - B*y)^2 + D*C^2*(y-y1)^2 = r1_sq*C^2
        # Let E = A - C*x1 = rhs - 2*dx*x1
        # (E - B*y)^2 + D*C^2*(y-y1)^2 = r1_sq*C^2
        # E^2 - 2*E*B*y + B^2*y^2 + D*C^2*(y^2 - 2*y*y1 + y1^2) = r1_sq*C^2
        # (B^2 + D*C^2)*y^2 - (2*E*B + 2*D*C^2*y1)*y + (E^2 + D*C^2*y1^2 - r1_sq*C^2) = 0
        
        E = A - C * x1
        a_coef = B*B + D*C*C
        b_coef = -(2*E*B + 2*D*C*C*y1)
        c_coef = E*E + D*C*C*y1*y1 - r1_sq*C*C
        
        # Discriminant
        disc = b_coef*b_coef - 4*a_coef*c_coef
        if disc < 0:
            return []
        
        # Check if disc is a perfect square (in rationals)
        if not _is_rational_dist_sq(disc, D):
            return []
        
        # sqrt(disc)
        num = disc.numerator
        den = disc.denominator
        sqrt_num = math.isqrt(num)
        sqrt_den = math.isqrt(den)
        sqrt_disc = Fraction(sqrt_num, sqrt_den)
        
        # y = (-b_coef ± sqrt_disc) / (2*a_coef)
        y1_val = (-b_coef + sqrt_disc) / (2 * a_coef)
        y2_val = (-b_coef - sqrt_disc) / (2 * a_coef)
        
        results = []
        for y_val in [y1_val, y2_val]:
            x_val = (A - B * y_val) / C
            # Verify the point satisfies both circle equations
            d1 = (x_val - x1)**2 + D*(y_val - y1)**2
            d2 = (x_val - x2)**2 + D*(y_val - y2)**2
            if d1 == r1_sq and d2 == r2_sq:
                results.append((x_val, y_val))
        
        return results
    
    # Case 2: dx == 0, so dy != 0 (circles have same x-center)
    elif dy != 0:
        # 2*D*dy*y = rhs
        # y = rhs / (2*D*dy)
        y_val = rhs / (2 * D * dy)
        
        # Substitute into first circle:
        # (x-x1)^2 + D*(y_val-y1)^2 = r1_sq
        # (x-x1)^2 = r1_sq - D*(y_val-y1)^2
        x_diff_sq = r1_sq - D*(y_val-y1)*(y_val-y1)
        if x_diff_sq < 0:
            return []
        
        if not _is_rational_dist_sq(x_diff_sq, D):
            return []
        
        num = x_diff_sq.numerator
        den = x_diff_sq.denominator
        sqrt_num = math.isqrt(num)
        sqrt_den = math.isqrt(den)
        sqrt_xdiff = Fraction(sqrt_num, sqrt_den)
        
        x1_val = x1 + sqrt_xdiff
        x2_val = x1 - sqrt_xdiff
        
        results = []
        for x_val in [x1_val, x2_val]:
            d1 = (x_val - x1)**2 + D*(y_val - y1)**2
            d2 = (x_val - x2)**2 + D*(y_val - y2)**2
            if d1 == r1_sq and d2 == r2_sq:
                results.append((x_val, y_val))
        
        return results
    
    return []

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Known 5-point set with D=1
    base_points = [(Fraction(0), Fraction(0)), 
                   (Fraction(-56), Fraction(33)), 
                   (Fraction(-16), Fraction(63)), 
                   (Fraction(0), Fraction(33)), 
                   (Fraction(16), Fraction(63))]
    D = 1
    
    best_points = base_points
    best_D = D
    
    max_time = 140.0
    
    # Strategy 1: Try to add a 6th point using circle intersections
    # For each pair of existing points, and various rational distances,
    # find intersection points and check if they have rational distance to all points.
    
    def try_add_point(pts, D, x, y):
        new_pt = (x, y)
        if new_pt in pts:
            return None
        new_pts = pts + [new_pt]
        if _validate(new_pts, D):
            return new_pts
        return None
    
    found_6 = False
    
    # Generate candidate points via circle intersections
    # For each pair (i, j), try various rational squared distances r1_sq, r2_sq
    n = len(base_points)
    candidates = set()
    
    # Small rational squared distances to try
    # Use perfect squares and small fractions
    test_r2_values = [Fraction(1), Fraction(4), Fraction(9), Fraction(16), Fraction(25),
                      Fraction(36), Fraction(49), Fraction(64), Fraction(81), Fraction(100),
                      Fraction(144), Fraction(169), Fraction(196), Fraction(225), Fraction(256),
                      Fraction(400), Fraction(625), Fraction(900), Fraction(1225), Fraction(1600)]
    
    for i in range(n):
        for j in range(i+1, n):
            if _now() > max_time:
                break
            p1 = base_points[i]
            p2 = base_points[j]
            
            # Try various distance pairs
            for r1_sq in test_r2_values:
                for r2_sq in test_r2_values:
                    if _now() > max_time:
                        break
                    # Check if circles can intersect
                    d2 = _dist_sq(p1, p2, D)
                    d = math.sqrt(float(d2)) if d2 > 0 else 0
                    r1 = math.sqrt(float(r1_sq))
                    r2 = math.sqrt(float(r2_sq))
                    
                    # Quick geometric check
                    if r1 + r2 < d - 1e-9 or abs(r1 - r2) > d + 1e-9:
                        continue
                    
                    # Find intersection
                    pts = _rational_circle_intersection(p1, p2, r1_sq, r2_sq, D)
                    for pt in pts:
                        # Check bounds
                        if abs(pt[0].numerator) <= 10**40 and abs(pt[0].denominator) <= 10**40:
                            if abs(pt[1].numerator) <= 10**40 and abs(pt[1].denominator) <= 10**40:
                                candidates.add(pt)
    
    # Also add some random candidates
    import random
    random.seed(42)
    for _ in range(500):
        if _now() > max_time:
            break
        x = Fraction(random.randint(-100, 100))
        y = Fraction(random.randint(-50, 150))
        candidates.add((x, y))
    
    # Test each candidate
    for cx, cy in candidates:
        if _now() > max_time:
            break
        new_pts = base_points + [(cx, cy)]
        if _validate(new_pts, D):
            best_points = new_pts
            found_6 = True
            break
    
    # If found 6, try to find 7
    if found_6 and _now() < max_time - 30:
        n = len(best_points)
        candidates7 = set()
        
        for i in range(n):
            for j in range(i+1, n):
                if _now() > max_time:
                    break
                p1 = best_points[i]
                p2 = best_points[j]
                
                for r1_sq in test_r2_values[:10]:
                    for r2_sq in test_r2_values[:10]:
                        if _now() > max_time:
                            break
                        d2 = _dist_sq(p1, p2, D)
                        d = math.sqrt(float(d2)) if d2 > 0 else 0
                        r1 = math.sqrt(float(r1_sq))
                        r2 = math.sqrt(float(r2_sq))
                        
                        if r1 + r2 < d - 1e-9 or abs(r1 - r2) > d + 1e-9:
                            continue
                        
                        pts = _rational_circle_intersection(p1, p2, r1_sq, r2_sq, D)
                        for pt in pts:
                            if abs(pt[0].numerator) <= 10**40 and abs(pt[0].denominator) <= 10**40:
                                if abs(pt[1].numerator) <= 10**40 and abs(pt[1].denominator) <= 10**40:
                                    candidates7.add(pt)
        
        for cx, cy in candidates7:
            if _now() > max_time:
                break
            new_pts = best_points + [(cx, cy)]
            if _validate(new_pts, D):
                best_points = new_pts
                break
    
    # Convert back to required format
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
