"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    """Check if integer n is a perfect square."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq_rat(p1, p2, D):
    """Compute squared distance between two points (x1,y1),(x2,y2) with metric D.
    Returns (num, den) of the squared distance, or None if not rational square."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    # d^2 = dx^2 + D * dy^2
    # dx, dy, D are integers (or Fractions)
    d2 = dx * dx + D * dy * dy
    if d2 < 0:
        return None
    # d2 is a Fraction. Check if it's a square of a rational.
    # d2 = num/den. sqrt(d2) rational iff num*den is a perfect square.
    num = d2.numerator
    den = d2.denominator
    prod = num * den
    if not _is_square(prod):
        return None
    return d2

def _check_collinear(p1, p2, p3):
    """Check if three points are collinear in the geometric space.
    Points are (x, y*sqrt(D)). Collinearity:
    (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3) == 0 (the sqrt(D) factors cancel)
    """
    cross = (p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])
    return cross == 0

def _check_concyclic(pts, D):
    """Check if 4 points are concyclic.
    Circle: (x-h)^2 + D(y-k)^2 = R^2
    => x^2 + D*y^2 + A*x + B*y + C = 0 where A=-2h, B=-2Dk, C=h^2+Dk^2-R^2
    For 4 points to be concyclic, the 4x4 system must have rank 3 (dependent).
    """
    # Use the determinant of the 4x4 matrix:
    # | x_i^2 + D*y_i^2  x_i  y_i  1 | for i=1..4
    # If det == 0, they are concyclic (or collinear, but we check that separately).
    
    def det4(m):
        # Laplace expansion along first row
        return (m[0][0] * (m[1][1]*m[2][2]*m[3][3] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][3]*m[2][2]*m[3][1])
                - m[0][1] * (m[1][0]*m[2][2]*m[3][3] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][3]*m[2][2]*m[3][0])
                + m[0][2] * (m[1][0]*m[2][1]*m[3][3] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][3]*m[2][1]*m[3][0])
                - m[0][3] * (m[1][0]*m[2][1]*m[3][2] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][2]*m[2][1]*m[3][0]))
    
    M = []
    for x, y in pts:
        M.append([x*x + D*y*y, x, y, 1])
    
    d = det4(M)
    if d == 0:
        # Check it's a real circle (R^2 > 0)
        # Solve for A, B, C using first 3 points
        M3 = M[:3]
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        
        det_M3 = det3(M3)
        if det_M3 == 0:
            return True  # degenerate, treat as concyclic to be safe
        
        # Cramer's rule for A, B, C
        # System: A*x_i + B*y_i + C = -(x_i^2 + D*y_i^2)
        v = [-(M[i][0]) for i in range(3)]
        
        M_A = [row[:] for row in M3]
        M_A[0][0] = v[0]
        M_A[1][0] = v[1]
        M_A[2][0] = v[2]
        A = Fraction(det3(M_A), det_M3)
        
        M_B = [row[:] for row in M3]
        M_B[0][1] = v[0]
        M_B[1][1] = v[1]
        M_B[2][1] = v[2]
        B = Fraction(det3(M_B), det_M3)
        
        M_C = [row[:] for row in M3]
        M_C[0][2] = v[0]
        M_C[1][2] = v[1]
        M_C[2][2] = v[2]
        C = Fraction(det3(M_C), det_M3)
        
        # R^2 = A^2/4 + B^2/(4D) - C
        R2 = A*A/4 + B*B/(4*D) - C
        if R2 > 0:
            return True
    return False

def _is_valid_set(points, D, max_time=None, start_time=None):
    """Check if a set of points is valid: distinct, rational distances, no 3 collinear, no 4 concyclic."""
    if max_time is not None and start_time is not None:
        if time.time() - start_time > max_time:
            return False
    
    n = len(points)
    if n < 3:
        return False
    
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False
    
    # Check rational distances
    for i in range(n):
        for j in range(i+1, n):
            if _dist_sq_rat(points[i], points[j], D) is None:
                return False
    
    # Check no 3 collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _check_collinear(points[i], points[j], points[k]):
                    return False
    
    # Check no 4 concyclic
    for combo in combinations(range(n), 4):
        pts = [points[i] for i in combo]
        if _check_concyclic(pts, D):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Known 5-point solution
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = points_5
    best_D = 1
    
    # Verify the known solution
    if _is_valid_set(points_5, 1, 30, start_time):
        best_points = points_5
        best_D = 1
    
    # Try to find a 6-point solution by incremental search
    # Strategy: start with 5 points, try to add a 6th point
    # Search space: lattice points with small coordinates
    
    max_coord = 100
    search_range = range(-max_coord, max_coord + 1)
    
    # Precompute all pairs with rational distances in the search space
    # This is expensive, so we do it incrementally
    
    found_6 = False
    points_to_try = [p for p in points_5]
    
    # Try adding points one at a time
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Try random lattice points
        import random
        if attempt == 0:
            random.seed(42)
        
        # Generate candidate points
        candidates = []
        for _ in range(50):
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            if (x, y) not in points_to_try:
                candidates.append((x, y))
        
        for cand in candidates:
            if time.time() > deadline:
                break
            
            new_points = points_to_try + [cand]
            if _is_valid_set(new_points, 1, 2, start_time):
                points_to_try = new_points
                found_6 = True
                break
        
        if found_6:
            break
    
    # Try a more systematic search for 6 points
    if len(points_to_try) < 6:
        # Use the known 5 points and try to find a 6th
        base = points_5
        for x in search_range:
            if time.time() > deadline:
                break
            for y in search_range:
                if time.time() > deadline:
                    break
                cand = (x, y)
                if cand in base:
                    continue
                new_points = base + [cand]
                if _is_valid_set(new_points, 1, 1, start_time):
                    points_to_try = new_points
                    break
            if len(points_to_try) >= 6:
                break
    
    # Try D=2, D=3 etc. with rational coordinates
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
        if time.time() > deadline:
            break
        
        # For D>1, points are (x, y*sqrt(D)) with rational x, y
        # We can search for small rational x, y
        # Start with integer x, y and check
        found = False
        for x1 in range(-20, 21):
            if time.time() > deadline:
                break
            for y1 in range(-20, 21):
                if time.time() > deadline:
                    break
                p1 = (Fraction(x1), Fraction(y1))
                for x2 in range(-20, 21):
                    if time.time() > deadline:
                        break
                    for y2 in range(-20, 21):
                        if time.time() > deadline:
                            break
                        p2 = (Fraction(x2), Fraction(y2))
                        if p1 == p2:
                            continue
                        # Check distance
                        if _dist_sq_rat(p1, p2, D) is None:
                            continue
                        for x3 in range(-20, 21):
                            if time.time() > deadline:
                                break
                            for y3 in range(-20, 21):
                                if time.time() > deadline:
                                    break
                                p3 = (Fraction(x3), Fraction(y3))
                                if p3 in [p1, p2]:
                                    continue
                                if _dist_sq_rat(p1, p3, D) is None:
                                    continue
                                if _dist_sq_rat(p2, p3, D) is None:
                                    continue
                                if _check_collinear(p1, p2, p3):
                                    continue
                                # Try to add a 4th point
                                for x4 in range(-20, 21):
                                    if time.time() > deadline:
                                        break
                                    for y4 in range(-20, 21):
                                        if time.time() > deadline:
                                            break
                                        p4 = (Fraction(x4), Fraction(y4))
                                        if p4 in [p1, p2, p3]:
                                            continue
                                        if _dist_sq_rat(p1, p4, D) is None:
                                            continue
                                        if _dist_sq_rat(p2, p4, D) is None:
                                            continue
                                        if _dist_sq_rat(p3, p4, D) is None:
                                            continue
                                        if _check_collinear(p1, p2, p4) or _check_collinear(p1, p3, p4) or _check_collinear(p2, p3, p4):
                                            continue
                                        if _check_concyclic([p1, p2, p3, p4], D):
                                            continue
                                        # Found 4 points. Try to extend to 5, 6...
                                        pts = [p1, p2, p3, p4]
                                        # Try adding more points
                                        for x5 in range(-20, 21):
                                            if time.time() > deadline:
                                                break
                                            for y5 in range(-20, 21):
                                                if time.time() > deadline:
                                                    break
                                                p5 = (Fraction(x5), Fraction(y5))
                                                if p5 in pts:
                                                    continue
                                                if any(_dist_sq_rat(p5, p, D) is None for p in pts):
                                                    continue
                                                if any(_check_collinear(p5, pts[i], pts[j]) for i in range(len(pts)) for j in range(i+1, len(pts))):
                                                    continue
                                                new_pts = pts + [p5]
                                                if any(_check_concyclic(combo, D) for combo in combinations(range(len(new_pts)), 4)):
                                                    continue
                                                pts = new_pts
                                                # Try to add 6th
                                                for x6 in range(-20, 21):
                                                    if time.time() > deadline:
                                                        break
                                                    for y6 in range(-20, 21):
                                                        if time.time() > deadline:
                                                            break
                                                        p6 = (Fraction(x6), Fraction(y6))
                                                        if p6 in pts:
                                                            continue
                                                        if any(_dist_sq_rat(p6, p, D) is None for p in pts):
                                                            continue
                                                        if any(_check_collinear(p6, pts[i], pts[j]) for i in range(len(pts)) for j in range(i+1, len(pts))):
                                                            continue
                                                        new_pts2 = pts + [p6]
                                                        if any(_check_concyclic(combo, D) for combo in combinations(range(len(new_pts2)), 4)):
                                                            continue
                                                        # Found 6 points with D
                                                        if len(pts) + 1 > len(points_to_try):
                                                            points_to_try = new_pts2
                                                            best_D = D
                                                        found = True
                                                        break
                                                    if found:
                                                        break
                                                if found:
                                                    break
                                                if len(pts) > len(points_to_try):
                                                    points_to_try = pts
                                                    best_D = D
                                                    found = True
                                                break
                                            if found:
                                                break
                                        if found:
                                            break
                                    if found:
                                        break
                            if found:
                                break
                    if found:
                        break
            if found:
                break
    
    # Final validation
    if not _is_valid_set(points_to_try, best_D, 30, start_time):
        # Fall back to known 5-point solution
        points_to_try = points_5
        best_D = 1
    
    # Convert to required format
    result_points = []
    for x, y in points_to_try:
        if isinstance(x, Fraction):
            x_num, x_den = x.numerator, x.denominator
        else:
            x_num, x_den = x, 1
        if isinstance(y, Fraction):
            y_num, y_den = y.numerator, y.denominator
        else:
            y_num, y_den = y, 1
        result_points.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': best_D, 'points': result_points}
# EVOLVE-BLOCK-END
