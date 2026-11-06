"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _is_squarefree(n):
    if n == 1:
        return True
    if n % 2 == 0:
        if n % 4 == 0:
            return False
        n //= 2
    i = 3
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 2
    return True

def _gcd_num(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a if a != 0 else 1

def _check_pairwise_rational(p1, p2, D):
    """Check if distance between p1 and p2 is rational."""
    x1n, x1d = p1[0]
    y1n, y1d = p1[1]
    x2n, x2d = p2[0]
    y2n, y2d = p2[1]
    
    # dx = x1 - x2, dy = y1 - y2
    dx_num = x1n * x2d - x2n * x1d
    dx_den = x1d * x2d
    
    dy_num = y1n * y2d - y2n * y1d
    dy_den = y1d * y2d
    
    # dist^2 = dx^2 + D * dy^2
    num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
    den = dx_den * dx_den * dy_den * dy_den
    
    if den == 0:
        return False
    if num < 0:
        return False
    
    g = _gcd_num(num, den)
    num_reduced = num // g
    den_reduced = den // g
    
    sqrt_num = int(math.isqrt(num_reduced))
    sqrt_den = int(math.isqrt(den_reduced))
    
    if sqrt_num * sqrt_num != num_reduced:
        return False
    if sqrt_den * sqrt_den != den_reduced:
        return False
    if sqrt_den == 0:
        return False
        
    return True

def _no_three_collinear(points, D):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi_n, xi_d = points[i][0]
                yi_n, yi_d = points[i][1]
                xj_n, xj_d = points[j][0]
                yj_n, yj_d = points[j][1]
                xk_n, xk_d = points[k][0]
                yk_n, yk_d = points[k][1]
                
                dx_ji_num = xj_n * xi_d - xi_n * xj_d
                dx_ji_den = xj_d * xi_d
                dy_ki_num = yk_n * yi_d - yi_n * yk_d
                dy_ki_den = yk_d * yi_d
                dx_ki_num = xk_n * xi_d - xi_n * xk_d
                dx_ki_den = xk_d * xi_d
                dy_ji_num = yj_n * yi_d - yi_n * yj_d
                dy_ji_den = yj_d * yi_d
                
                cross_num2 = dx_ji_num * dy_ki_num * dx_ki_den * dy_ji_den - dx_ki_num * dy_ji_num * dx_ji_den * dy_ki_den
                cross_den2 = dx_ji_den * dy_ki_den * dx_ki_den * dy_ji_den
                
                if cross_den2 == 0:
                    continue
                if cross_num2 != 0:
                    continue
                return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    xi_n, xi_d = points[i][0]
                    yi_n, yi_d = points[i][1]
                    xj_n, xj_d = points[j][0]
                    yj_n, yj_d = points[j][1]
                    xk_n, xk_d = points[k][0]
                    yk_n, yk_d = points[k][1]
                    xl_n, xl_d = points[l][0]
                    yl_n, yl_d = points[l][1]
                    
                    xi = Fraction(xi_n, xi_d)
                    yi = Fraction(yi_n, yi_d)
                    xj = Fraction(xj_n, xj_d)
                    yj = Fraction(yj_n, yj_d)
                    xk = Fraction(xk_n, xk_d)
                    yk = Fraction(yk_n, yk_d)
                    xl = Fraction(xl_n, xl_d)
                    yl = Fraction(yl_n, yl_d)
                    
                    A1 = xj - xi
                    B1 = yj - yi
                    C1 = -(xj*xj + D*yj*yj) + (xi*xi + D*yi*yi)
                    
                    A2 = xk - xi
                    B2 = yk - yi
                    C2 = -(xk*xk + D*yk*yk) + (xi*xi + D*yi*yi)
                    
                    det = A1*B2 - A2*B1
                    if det == 0:
                        continue
                    
                    a = (C1*B2 - C2*B1) / det
                    b_prime = (A1*C2 - A2*C1) / det
                    c = -(xi*xi + D*yi*yi) - a*xi - b_prime*yi
                    
                    if xl*xl + D*yl*yl + a*xl + b_prime*yl + c != 0:
                        continue
                    return False
    return True

def _validate(points, D):
    """Validate the full configuration."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    for i in range(n):
        for j in range(i+1, n):
            if not _check_pairwise_rational(points[i], points[j], D):
                return False
    if not _no_three_collinear(points, D):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _find_6_point_set(D=1, time_limit=30.0):
    """
    Search for a 6-point integral point set.
    Strategy: Try configurations on two parallel lines, and also try
    extending the known 5-point solution.
    """
    start = time.time()
    
    # First, try to extend the known 5-point solution
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try adding a 6th point near the existing configuration
    # The existing points span x from -56 to 16, y from 0 to 63
    # Try points on the same lines or nearby
    
    # Check which lines the known points are on
    # (0,0) and (0,33) are on x=0
    # (-56,33) and (-16,33) and (16,33) are on y=33
    # (-16,63) and (16,63) are on y=63
    
    # Try to add a point on y=0, y=33, or y=63 that works with all 5
    for y_line in [0, 33, 63]:
        if time.time() - start > time_limit:
            break
        for x in range(-200, 201):
            # Skip if already in the set
            if (x, y_line) in known_points:
                continue
            
            new_pt = [[x, 1], [y_line, 1]]
            candidate = [[[px, 1], [py, 1]] for px, py in known_points] + [new_pt]
            
            # Quick pairwise rational distance check before full validation
            valid = True
            for i in range(5):
                if not _check_pairwise_rational(candidate[i], new_pt, D):
                    valid = False
                    break
            
            if valid and _validate(candidate, D):
                return candidate
    
    # Now try the two-line approach with wider search
    for h in range(1, 50):
        if time.time() - start > time_limit:
            break
        
        # Find all valid dx values where dx^2 + h^2 is a perfect square
        valid_dx = set()
        max_k = 2000
        for k in range(h + 1, max_k):
            dx2 = k * k - h * h
            dx = int(math.isqrt(dx2))
            if dx * dx == dx2 and dx > 0:
                valid_dx.add(dx)
                valid_dx.add(-dx)
        
        if len(valid_dx) < 6:
            continue
        
        valid_dx_sorted = sorted(valid_dx)
        top_candidates = [dx for dx in valid_dx_sorted if abs(dx) <= 500]
        
        if len(top_candidates) < 3:
            continue
        
        # Try combinations of 3 top points
        for idx in range(len(top_candidates)):
            if time.time() - start > time_limit:
                break
            t1 = top_candidates[idx]
            for idx2 in range(idx + 1, len(top_candidates)):
                t2 = top_candidates[idx2]
                for idx3 in range(idx2 + 1, len(top_candidates)):
                    t3 = top_candidates[idx3]
                    
                    set1 = {t1 + dx for dx in valid_dx}
                    set2 = {t2 + dx for dx in valid_dx}
                    set3 = {t3 + dx for dx in valid_dx}
                    
                    possible_x = set1 & set2 & set3
                    possible_x = [x for x in possible_x if abs(x) <= 500 and x != 0]
                    
                    if len(possible_x) < 2:
                        continue
                    
                    for x2, x3 in combinations(possible_x, 2):
                        if x2 == x3:
                            continue
                        
                        pts = [
                            [[0, 1], [0, 1]],
                            [[x2, 1], [0, 1]],
                            [[x3, 1], [0, 1]],
                            [[t1, 1], [h, 1]],
                            [[t2, 1], [h, 1]],
                            [[t3, 1], [h, 1]]
                        ]
                        
                        if len(set(tuple(tuple(c) for c in p) for p in pts)) < 6:
                            continue
                        
                        if _validate(pts, D):
                            return pts
    return None

def _find_7_point_set(D=1, time_limit=20.0):
    """
    Try to extend a 6-point set to 7 points.
    """
    start = time.time()
    
    # First find a 6-point set
    pts6 = _find_6_point_set(D, time_limit=8.0)
    if not pts6:
        return None
    
    # Extract y-coordinates used
    y_vals = set()
    for p in pts6:
        y_vals.add(p[1][0] * 1 // p[1][1])
    
    # Try adding a 7th point on existing lines or new lines
    # Try x-coordinates on each existing y-line
    for y_line in y_vals:
        for x in range(-200, 201):
            if time.time() - start > time_limit:
                break
            new_pt = [[x, 1], [y_line, 1]]
            
            # Check if this point is already in the set
            if any(p[0][0] == x and p[0][1] == 1 and p[1][0] == y_line and p[1][1] == 1 for p in pts6):
                continue
            
            # Quick pairwise check
            valid = True
            for p in pts6:
                if not _check_pairwise_rational(p, new_pt, D):
                    valid = False
                    break
            
            if not valid:
                continue
            
            candidate = pts6 + [new_pt]
            if _validate(candidate, D):
                return candidate
    
    # Try new y-lines
    for y_line in range(1, 100):
        if time.time() - start > time_limit:
            break
        if y_line in y_vals:
            continue
        for x in range(-100, 101):
            new_pt = [[x, 1], [y_line, 1]]
            
            valid = True
            for p in pts6:
                if not _check_pairwise_rational(p, new_pt, D):
                    valid = False
                    break
            
            if not valid:
                continue
            
            candidate = pts6 + [new_pt]
            if _validate(candidate, D):
                return candidate
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_result = None
    best_score = 0
    
    # Start with the known 5-point solution
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
    best_score = 5
    
    deadline = _START_TIME + 140.0
    
    # Try to find a 6-point set with generous time
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 30
        pts6 = _find_6_point_set(D=1, time_limit=remaining)
        if pts6:
            if _validate(pts6, 1):
                best_score = 6
                best_result = {'D': 1, 'points': pts6}
        
        # If we found 6, try to push for 7
        if best_score == 6 and time.time() < deadline - 15:
            remaining = deadline - time.time() - 15
            pts7 = _find_7_point_set(D=1, time_limit=remaining)
            if pts7:
                if _validate(pts7, 1):
                    best_score = 7
                    best_result = {'D': 1, 'points': pts7}
    
    # Final validation
    if best_result is None:
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
        
    return best_result

# EVOLVE-BLOCK-END
