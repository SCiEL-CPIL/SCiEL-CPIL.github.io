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
                dy_ki_num = yk_n * yi_d - yi_n * yk_d
                dx_ki_num = xk_n * xi_d - xi_n * xk_d
                dy_ji_num = yj_n * yi_d - yi_n * yj_d
                
                # Cross product: (xj-xi)*(yk-yi) - (xk-xi)*(yj-yi)
                cross_num = dx_ji_num * dy_ki_num * yi_d * yi_d - dx_ki_num * dy_ji_num * yi_d * yi_d
                if cross_num != 0:
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

def _find_6_point_set(D=1, search_limit=200, time_limit=10.0):
    """Search for a 6-point set on two lines y=0 and y=h."""
    start = time.time()
    # Try various heights h
    for h in range(1, 80):
        if time.time() > start + time_limit:
            break
        # Find all x-differences such that dx^2 + h^2 is a perfect square
        valid_dx = set()
        for k in range(h+1, int(math.sqrt(search_limit*search_limit + h*h)) + 2):
            dx2 = k*k - h*h
            if dx2 < 0:
                break
            dx = int(math.isqrt(dx2))
            if dx*dx == dx2 and dx > 0:
                valid_dx.add(dx)
                valid_dx.add(-dx)
        
        if len(valid_dx) < 4:
            continue
            
        # We need to choose 3 points on y=0 and 3 on y=h
        # Fix one bottom point at x=0
        # Top points must be at positions in valid_dx from 0
        candidates_top = [dx for dx in valid_dx if abs(dx) <= search_limit]
        if len(candidates_top) < 3:
            continue
            
        # Try combinations of 3 top points
        for y1, y2, y3 in combinations(sorted(candidates_top), 3):
            if time.time() > start + time_limit:
                break
            
            # Find x2, x3 on bottom line such that distances to all top points are rational
            # x2 - y1, x2 - y2, x2 - y3 must be in valid_dx
            possible_x2 = set()
            for dx in valid_dx:
                x2_candidate = y1 + dx
                if (x2_candidate - y2) in valid_dx and (x2_candidate - y3) in valid_dx:
                    if abs(x2_candidate) <= search_limit:
                        possible_x2.add(x2_candidate)
            
            possible_x3 = set()
            for dx in valid_dx:
                x3_candidate = y1 + dx
                if (x3_candidate - y2) in valid_dx and (x3_candidate - y3) in valid_dx:
                    if abs(x3_candidate) <= search_limit:
                        possible_x3.add(x3_candidate)
            
            # Also check from y2
            for dx in valid_dx:
                x2_candidate = y2 + dx
                if (x2_candidate - y1) in valid_dx and (x2_candidate - y3) in valid_dx:
                    if abs(x2_candidate) <= search_limit:
                        possible_x2.add(x2_candidate)
            
            for dx in valid_dx:
                x3_candidate = y2 + dx
                if (x3_candidate - y1) in valid_dx and (x3_candidate - y3) in valid_dx:
                    if abs(x3_candidate) <= search_limit:
                        possible_x3.add(x3_candidate)
            
            # Also check from y3
            for dx in valid_dx:
                x2_candidate = y3 + dx
                if (x2_candidate - y1) in valid_dx and (x2_candidate - y2) in valid_dx:
                    if abs(x2_candidate) <= search_limit:
                        possible_x2.add(x2_candidate)
            
            for dx in valid_dx:
                x3_candidate = y3 + dx
                if (x3_candidate - y1) in valid_dx and (x3_candidate - y2) in valid_dx:
                    if abs(x3_candidate) <= search_limit:
                        possible_x3.add(x3_candidate)
            
            # Filter out x=0 (already used)
            possible_x2.discard(0)
            possible_x3.discard(0)
            
            if len(possible_x2) < 2:
                continue
                
            # Pick two distinct x values for bottom points
            for x2, x3 in combinations(sorted(possible_x2), 2):
                if x2 == x3:
                    continue
                    
                pts = [
                    [[0, 1], [0, 1]],
                    [[x2, 1], [0, 1]],
                    [[x3, 1], [0, 1]],
                    [[y1, 1], [h, 1]],
                    [[y2, 1], [h, 1]],
                    [[y3, 1], [h, 1]]
                ]
                
                # Quick check: distinct points
                if len(set(tuple(tuple(coord) for coord in p) for p in pts)) < 6:
                    continue
                    
                if _validate(pts, D):
                    return pts
    return None

def _find_7_point_set(D=1, search_limit=150, time_limit=20.0):
    """Search for a 7-point set on two lines y=0 and y=h with 4 on one line, 3 on other."""
    start = time.time()
    for h in range(1, 60):
        if time.time() > start + time_limit:
            break
        valid_dx = set()
        for k in range(h+1, int(math.sqrt(search_limit*search_limit + h*h)) + 2):
            dx2 = k*k - h*h
            if dx2 < 0:
                break
            dx = int(math.isqrt(dx2))
            if dx*dx == dx2 and dx > 0:
                valid_dx.add(dx)
                valid_dx.add(-dx)
        
        if len(valid_dx) < 6:
            continue
            
        candidates = [dx for dx in valid_dx if abs(dx) <= search_limit]
        if len(candidates) < 4:
            continue
        
        # Try 4 points on bottom (y=0), 3 on top (y=h)
        # Fix bottom points at 0, a, b, c
        # Top points at positions t1, t2, t3
        # All cross distances must be in valid_dx
        
        # This is complex; try a simpler approach
        # Fix bottom: 0, d1, d2, d3 where d1,d2,d3 are from valid_dx
        # Top: t1, t2, t3 where ti - 0, ti - d1, etc. are in valid_dx
        
        for d1 in candidates[:10]:
            if time.time() > start + time_limit:
                break
            if d1 == 0:
                continue
            for d2 in candidates[:10]:
                if time.time() > start + time_limit:
                    break
                if d2 == 0 or d2 == d1:
                    continue
                for d3 in candidates[:10]:
                    if time.time() > start + time_limit:
                        break
                    if d3 == 0 or d3 == d1 or d3 == d2:
                        continue
                    
                    # Bottom points: 0, d1, d2, d3
                    # Find top points t such that t, t-d1, t-d2, t-d3 all in valid_dx
                    possible_t = set()
                    for dx in valid_dx:
                        t_candidate = dx  # t - 0 = dx
                        if (t_candidate - d1) in valid_dx and (t_candidate - d2) in valid_dx and (t_candidate - d3) in valid_dx:
                            if abs(t_candidate) <= search_limit:
                                possible_t.add(t_candidate)
                    
                    if len(possible_t) < 3:
                        continue
                    
                    for t1, t2, t3 in combinations(sorted(possible_t), 3):
                        if time.time() > start + time_limit:
                            break
                        pts = [
                            [[0, 1], [0, 1]],
                            [[d1, 1], [0, 1]],
                            [[d2, 1], [0, 1]],
                            [[d3, 1], [0, 1]],
                            [[t1, 1], [h, 1]],
                            [[t2, 1], [h, 1]],
                            [[t3, 1], [h, 1]]
                        ]
                        
                        if len(set(tuple(tuple(coord) for coord in p) for p in pts)) < 7:
                            continue
                            
                        if _validate(pts, D):
                            return pts
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
    
    # Try to find a 6-point set using the two-line heuristic
    if time.time() < deadline - 30:
        pts6 = _find_6_point_set(D=1, search_limit=200, time_limit=min(30, deadline - time.time() - 20))
        if pts6:
            if _validate(pts6, 1):
                best_score = 6
                best_result = {'D': 1, 'points': pts6}
                print("Found 6 points")
                
    # If we found 6, try to push for 7
    if best_score >= 6 and time.time() < deadline - 15:
        pts7 = _find_7_point_set(D=1, search_limit=150, time_limit=min(25, deadline - time.time() - 10))
        if pts7:
            if _validate(pts7, 1):
                best_score = 7
                best_result = {'D': 1, 'points': pts7}
                print("Found 7 points")
    
    # Try D=2 as well
    if best_score < 7 and time.time() < deadline - 15:
        pts6_d2 = _find_6_point_set(D=2, search_limit=150, time_limit=min(15, deadline - time.time() - 10))
        if pts6_d2:
            if _validate(pts6_d2, 2):
                best_score = max(best_score, 6)
                best_result = {'D': 2, 'points': pts6_d2}
                print("Found 6 points with D=2")
    
    # Final validation
    if best_result is None:
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
        
    return best_result

# EVOLVE-BLOCK-END
