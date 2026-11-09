import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def is_square(n):
        if n < 0:
            return False
        s = math.isqrt(n)
        return s * s == n

    def check_rational_dist(dx_num, dx_den, dy_num, dy_den, D):
        """Check if (dx_num/dx_den)^2 + D*(dy_num/dy_den)^2 is a perfect square of a rational."""
        num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
        den = dx_den * dx_den * dy_den * dy_den
        if den == 0:
            return False
        if num == 0:
            return True
        prod = num * den
        if prod < 0:
            return False
        return is_square(prod)

    def check_general_position(points, D):
        """Check no three collinear and no four concyclic. Points are (x_num, x_den, y_num, y_den)."""
        n = len(points)
        
        def get_frac(i):
            p = points[i]
            return Fraction(p[0], p[1]), Fraction(p[2], p[3])
        
        # Check no three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = get_frac(i)
                    x2, y2 = get_frac(j)
                    x3, y3 = get_frac(k)
                    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                    if cross == 0:
                        return False
        
        # Check no four concyclic
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        
                        x1 = Fraction(p1[0], p1[1])
                        y1 = Fraction(p1[2], p1[3])
                        x2 = Fraction(p2[0], p2[1])
                        y2 = Fraction(p2[2], p2[3])
                        x3 = Fraction(p3[0], p3[1])
                        y3 = Fraction(p3[2], p3[3])
                        x4 = Fraction(p4[0], p4[1])
                        y4 = Fraction(p4[2], p4[3])
                        
                        v1 = x1 * x1 + D * y1 * y1
                        v2 = x2 * x2 + D * y2 * y2
                        v3 = x3 * x3 + D * y3 * y3
                        v4 = x4 * x4 + D * y4 * y4
                        
                        a11 = v1 - v4
                        a12 = x1 - x4
                        a13 = y1 - y4
                        a21 = v2 - v4
                        a22 = x2 - x4
                        a23 = y2 - y4
                        a31 = v3 - v4
                        a32 = x3 - x4
                        a33 = y3 - y4
                        
                        det3 = (a11 * (a22 * a33 - a23 * a32)
                               - a12 * (a21 * a33 - a23 * a31)
                               + a13 * (a21 * a32 - a22 * a31))
                        
                        if det3 == 0:
                            return False
        
        return True

    def is_valid_set(points, D):
        """Full validity check."""
        n = len(points)
        if n < 3:
            return False
        
        # Check distinctness
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                if Fraction(p1[0], p1[1]) == Fraction(p2[0], p2[1]) and \
                   Fraction(p1[2], p1[3]) == Fraction(p2[2], p2[3]):
                    return False
        
        # Check rational distances
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                x1 = Fraction(p1[0], p1[1])
                y1 = Fraction(p1[2], p1[3])
                x2 = Fraction(p2[0], p2[1])
                y2 = Fraction(p2[2], p2[3])
                dx = x2 - x1
                dy = y2 - y1
                if not check_rational_dist(dx.numerator, dx.denominator, dy.numerator, dy.denominator, D):
                    return False
        
        # Check general position
        if not check_general_position(points, D):
            return False
        
        return True

    def try_add_point(points, D, max_coord_bound):
        """Try to add a 6th point to a 5-point set by searching nearby integer points."""
        if len(points) != 5:
            return None
        
        xs = [Fraction(p[0], p[1]) for p in points]
        ys = [Fraction(p[2], p[3]) for p in points]
        
        min_x = min(int(math.floor(x)) for x in xs)
        max_x = max(int(math.ceil(x)) for x in xs)
        min_y = min(int(math.floor(y)) for y in ys)
        max_y = max(int(math.ceil(y)) for y in ys)
        
        expand = max_coord_bound
        search_min_x = min_x - expand
        search_max_x = max_x + expand
        search_min_y = min_y - expand
        search_max_y = max_y + expand
        
        max_points_to_try = 50000
        
        existing = set()
        for p in points:
            existing.add((Fraction(p[0], p[1]), Fraction(p[2], p[3])))
        
        count = 0
        for x in range(search_min_x, search_max_x + 1):
            for y in range(search_min_y, search_max_y + 1):
                if count >= max_points_to_try:
                    return None
                count += 1
                
                new_point = (x, 1, y, 1)
                if (Fraction(x, 1), Fraction(y, 1)) in existing:
                    continue
                
                valid = True
                for i in range(5):
                    p = points[i]
                    px = Fraction(p[0], p[1])
                    py = Fraction(p[2], p[3])
                    dx = Fraction(x, 1) - px
                    dy = Fraction(y, 1) - py
                    if not check_rational_dist(dx.numerator, dx.denominator, dy.numerator, dy.denominator, D):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_points = points + [new_point]
                if check_general_position(new_points, D):
                    return new_point
        
        return None

    # ===== Strategy: Hexagon in Q(sqrt(3)) with rational perturbation =====
    D_hex = 3
    
    # 6 vertices of a regular hexagon centered at origin with "radius" 1:
    # In Q(sqrt(3)), the 6th roots of unity scaled: (cos(k*pi/3), sin(k*pi/3))
    # cos(0)=1, sin(0)=0 -> (1, 0)
    # cos(pi/3)=1/2, sin(pi/3)=sqrt(3)/2 -> (1/2, 1/2) [y*sqrt(3)]
    # cos(2pi/3)=-1/2, sin(2pi/3)=sqrt(3)/2 -> (-1/2, 1/2)
    # cos(pi)=-1, sin(pi)=0 -> (-1, 0)
    # cos(4pi/3)=-1/2, sin(4pi/3)=-sqrt(3)/2 -> (-1/2, -1/2)
    # cos(5pi/3)=1/2, sin(5pi/3)=-sqrt(3)/2 -> (1/2, -1/2)
    
    # Verify: distance between adjacent vertices, e.g., (1,0) and (1/2, 1/2):
    # dx = -1/2, dy = 1/2
    # dist^2 = (1/2)^2 + 3*(1/2)^2 = 1/4 + 3/4 = 1. Rational! Good.
    # Distance (1,0) to (-1/2, 1/2): dx=-3/2, dy=1/2
    # dist^2 = 9/4 + 3/4 = 12/4 = 3. NOT a rational square (sqrt(3)).
    # So not all pairwise distances are rational in the regular hexagon!
    
    # We need a different approach. Let's use a known integral point set structure
    # and work in Q(sqrt(3)) to get more flexibility.
    
    # Alternative: Use the known 5-point integral set and try to find a 6th point
    # more aggressively, or construct a 6-point set directly in Q(sqrt(3)).
    
    # Let's try: points on a circle in Q(sqrt(3)) that are rational-distance apart.
    # A circle x^2 + 3y^2 = R^2 with rational points.
    # Parametrize: for a circle x^2 + D*y^2 = 1, rational points exist.
    # x = (1 - D*t^2)/(1 + D*t^2), y = 2*t/(1 + D*t^2) for rational t.
    # With D=3: x = (1-3t^2)/(1+3t^2), y = 2t/(1+3t^2)
    # Distance between two such points (t1, t2):
    # This is complex. Let's try a direct search.
    
    # Direct search for 6 points in Q(sqrt(3)) with small integer coords
    def search_6_points_D3():
        """Search for 6 points with integer x,y in Q(sqrt(3)) with rational pairwise distances."""
        # Points: (x, y) meaning (x, y*sqrt(3))
        # dist^2 between (x1,y1) and (x2,y2) = (x1-x2)^2 + 3*(y1-y2)^2
        # Need this to be a perfect square of a rational (integer if coords are integers)
        
        # Generate candidate points with small integer coords
        # Check pairwise: (dx)^2 + 3*(dy)^2 must be a perfect square
        
        def is_valid_pair(p1, p2):
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            val = dx*dx + 3*dy*dy
            if val < 0:
                return False
            s = math.isqrt(val)
            return s*s == val
        
        # Try all combinations of 6 points from a grid
        # Grid: x in [-10, 10], y in [-10, 10]
        candidates = []
        for x in range(-10, 11):
            for y in range(-10, 11):
                candidates.append((x, y))
        
        # First, find all pairs that have rational distance
        # Then find 6-cliques in the compatibility graph
        
        n_cand = len(candidates)
        # Build adjacency: cand[i] compatible with cand[j] if is_valid_pair
        # For efficiency, precompute
        
        # Actually, let's be smarter. Start with a known good 5-point set and extend.
        # Or do a backtracking search.
        
        # Backtracking: build set incrementally
        def backtrack(current_set, idx):
            if len(current_set) == 6:
                # Check general position
                pts = [(c[0], 1, c[1], 1) for c in current_set]
                if check_general_position(pts, 3):
                    return current_set
                return None
            
            remaining = deadline - time.time()
            if remaining < 5:
                return None
            
            for i in range(idx, n_cand):
                cand = candidates[i]
                # Check compatibility with all in current_set
                ok = True
                for c in current_set:
                    if not is_valid_pair(cand, c):
                        ok = False
                        break
                if not ok:
                    continue
                
                current_set.append(cand)
                result = backtrack(current_set, i + 1)
                current_set.pop()
                if result is not None:
                    return result
            
            return None
        
        return backtrack([], 0)
    
    # Try the hexagon-perturbation approach first
    # Take 5 of the 6 hexagon vertices (which have rational distances among themselves
    # if we pick the right subset), then add a 6th nearby point.
    
    # Actually, let's verify which subsets of hexagon vertices have all rational distances.
    # Hex vertices (x, y*sqrt(3)):
    # H0: (1, 0)
    # H1: (1/2, 1/2)  
    # H2: (-1/2, 1/2)
    # H3: (-1, 0)
    # H4: (-1/2, -1/2)
    # H5: (1/2, -1/2)
    
    # Pairwise dist^2 (with D=3):
    # H0-H1: (1/2)^2 + 3(1/2)^2 = 1/4+3/4=1 ✓
    # H0-H2: (3/2)^2 + 3(1/2)^2 = 9/4+3/4=3 ✗ (not a square)
    # H0-H3: 2^2 + 0 = 4 ✓
    # H0-H4: (3/2)^2 + 3(1/2)^2 = 3 ✗
    # H0-H5: (1/2)^2 + 3(1/2)^2 = 1 ✓
    # H1-H2: 1^2 + 0 = 1 ✓
    # H1-H3: (3/2)^2 + 3(1/2)^2 = 3 ✗
    # H1-H4: 1^2 + 3(1)^2 = 4 ✓
    # H1-H5: 0 + 3(1)^2 = 3 ✗
    # H2-H3: (1/2)^2 + 3(1/2)^2 = 1 ✓
    # H2-H4: 0 + 3(1)^2 = 3 ✗
    # H2-H5: (1)^2 + 3(1)^2 = 4 ✓
    # H3-H4: (1/2)^2 + 3(1/2)^2 = 1 ✓
    # H3-H5: (3/2)^2 + 3(1/2)^2 = 3 ✗
    # H4-H5: 1^2 + 0 = 1 ✓
    
    # So valid pairs (dist^2 is a perfect square): 
    # H0-H1(1), H0-H3(4), H0-H5(1), H1-H2(1), H1-H4(4), H2-H3(1), H2-H5(4), H3-H4(1), H4-H5(1)
    # 
    # Need 6 points all mutually connected. Let's check if any 6-subset works.
    # Since we only have 6 vertices, we need ALL pairs to be valid. But H0-H2 is not valid.
    # So the full hexagon doesn't work.
    
    # Try subsets of 5:
    # {H0,H1,H3,H4,H5}: H0-H1✓, H0-H3✓, H0-H5✓, H1-H3✗ -> No
    # {H0,H1,H2,H4,H5}: H0-H2✗ -> No  
    # {H0,H1,H2,H3,H5}: H0-H2✗, H1-H3✗ -> No
    # {H0,H1,H2,H3,H4}: H0-H2✗, H1-H3✗, H2-H4✗ -> No
    # {H0,H1,H2,H4,H5}: already checked
    # {H0,H1,H3,H4,H5}: already checked
    # {H0,H2,H3,H4,H5}: H0-H2✗ -> No
    # {H1,H2,H3,H4,H5}: H1-H3✗, H1-H5✗, H2-H4✗ -> No
    
    # No 5-subset of the hexagon works! The hexagon approach for 5+ points fails.
    
    # Let's try the direct backtracking search
    if time_left() > 30:
        result_6 = search_6_points_D3()
        if result_6 is not None:
            # Convert to format
            pts_formatted = [(c[0], 1, c[1], 1) for c in result_6]
            if is_valid_set(pts_formatted, 3):
                final_points = pts_formatted
                final_D = 3
                result_points = []
                for p in final_points:
                    x_frac = Fraction(p[0], p[1])
                    y_frac = Fraction(p[2], p[3])
                    result_points.append([[x_frac.numerator, x_frac.denominator], [y_frac.numerator, y_frac.denominator]])
                return {'D': final_D, 'points': result_points}
    
    # Fallback: use the known 5-point integral set
    points_5 = [
        (0, 1, 0, 1),
        (-56, 1, 33, 1),
        (-16, 1, 63, 1),
        (0, 1, 33, 1),
        (16, 1, 63, 1)
    ]
    
    D = 1
    
    best_points = points_5
    
    # Try to find a 6th point with D=1
    if time_left() > 15:
        p6 = try_add_point(points_5, D, 200)
        if p6 is not None:
            candidate = points_5 + [p6]
            if is_valid_set(candidate, D):
                best_points = candidate
    
    # Try randomized search for a 6th point with D=1, larger range
    if len(best_points) == 5 and time_left() > 20:
        random.seed(42)
        base = points_5
        bx = [Fraction(p[0], p[1]) for p in base]
        by = [Fraction(p[2], p[3]) for p in base]
        cx = sum(bx) / len(bx)
        cy = sum(by) / len(by)
        
        max_tries = 100000
        for trial in range(max_tries):
            if time_left() < 5:
                break
            x = int(random.gauss(int(cx), 300))
            y = int(random.gauss(int(cy), 300))
            
            new_point = (x, 1, y, 1)
            
            distinct = True
            for p in base:
                if Fraction(p[0], p[1]) == Fraction(x, 1) and Fraction(p[2], p[3]) == Fraction(y, 1):
                    distinct = False
                    break
            if not distinct:
                continue
            
            valid = True
            for p in base:
                px = Fraction(p[0], p[1])
                py = Fraction(p[2], p[3])
                dx = Fraction(x, 1) - px
                dy = Fraction(y, 1) - py
                if not check_rational_dist(dx.numerator, dx.denominator, dy.numerator, dy.denominator, D):
                    valid = False
                    break
            if not valid:
                continue
            
            new_points = base + [new_point]
            if check_general_position(new_points, D):
                best_points = new_points
                break
    
    # Final validation
    final_D = D
    final_points = best_points
    
    result_points = []
    for p in final_points:
        x_frac = Fraction(p[0], p[1])
        y_frac = Fraction(p[2], p[3])
        result_points.append([[x_frac.numerator, x_frac.denominator], [y_frac.numerator, y_frac.denominator]])
    
    return {'D': final_D, 'points': result_points}
