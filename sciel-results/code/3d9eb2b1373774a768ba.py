"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from math import isqrt, gcd
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_squarefree(n):
    if n <= 1:
        return True
    n = abs(n)
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n > 1:
        r = isqrt(n)
        if r * r == n:
            return False
    return True

def _dist_sq(p1, p2, D):
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _is_rational_dist(p1, p2, D):
    d2 = _dist_sq(p1, p2, D)
    return _is_square(d2.numerator) and _is_square(d2.denominator)

def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _concyclic(p1, p2, p3, p4, D):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    def det3(xa, ya, xb, yb, xc, yc):
        return xa*(yb - yc) - ya*(xb - xc) + (xb*yc - xc*yb)
    
    A = x1*x1 + D*y1*y1
    B = x2*x2 + D*y2*y2
    C = x3*x3 + D*y3*y3
    E = x4*x4 + D*y4*y4
    
    det = (A * det3(x2, y2, x3, y3, x4, y4)
           - B * det3(x1, y1, x3, y3, x4, y4)
           + C * det3(x1, y1, x2, y2, x4, y4)
           - E * det3(x1, y1, x2, y2, x3, y3))
    
    return det == 0

def _validate(points, D):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False
            if not _is_rational_dist(points[i], points[j], D):
                return False
    for i, j, k in combinations(range(n), 3):
        if _collinear(points[i], points[j], points[k]):
            return False
    for i, j, k, l in combinations(range(n), 4):
        if _concyclic(points[i], points[j], points[k], points[l], D):
            return False
    return True

def _try_known_6point_sets():
    """Try several known 6-point rational distance sets."""
    # Known 6-point sets with D=1 (all integer coordinates)
    # Set 1: From literature
    sets = [
        # (x1,y1), (x2,y2), ..., (x6,y6)
        [(0,0), (7,24), (15,8), (24,7), (8,15), (24,0)],
        [(0,0), (5,12), (12,5), (13,0), (0,13), (5,0)],
        [(0,0), (3,4), (4,3), (5,0), (0,5), (3,0)],
        # Try some other configurations
        [(0,0), (9,12), (12,9), (15,0), (0,15), (9,0)],
        [(0,0), (20,21), (21,20), (29,0), (0,29), (20,0)],
        # Another family
        [(0,0), (65,72), (72,65), (97,0), (0,97), (65,0)],
    ]
    
    for pts in sets:
        frac_pts = [(Fraction(x), Fraction(y)) for x, y in pts]
        if _validate(frac_pts, 1):
            return frac_pts, 1
    return None, None

def _search_extension(base_set, D, max_extra=3, time_limit=60):
    """Try to extend a base set by finding additional points."""
    start_time = time.time()
    current_set = list(base_set)
    
    # Generate many candidate points
    candidates = []
    
    # Points on circles centered at various base points
    for bp in current_set:
        bx, by = bp
        for r_num in range(1, 50):
            for r_den in [1, 2, 3, 4, 5]:
                r = Fraction(r_num, r_den)
                # Parameterize circle of radius r centered at (bx, by)
                for t_num in range(-30, 31):
                    for t_den in range(1, 11):
                        if gcd(t_num, t_den) != 1:
                            continue
                        t = Fraction(t_num, t_den)
                        denom = 1 + D * t * t
                        if denom == 0:
                            continue
                        x = bx + r * (1 - D * t * t) / denom
                        y = by + r * 2 * t / denom
                        
                        if abs(x.numerator) > 10**20 or abs(x.denominator) > 10**20:
                            continue
                        if abs(y.numerator) > 10**20 or abs(y.denominator) > 10**20:
                            continue
                        
                        candidates.append((x, y))
    
    # Also add grid points for D=1
    if D == 1:
        for x in range(-100, 101):
            for y in range(-100, 101):
                candidates.append((Fraction(x), Fraction(y)))
    
    # Remove duplicates and points already in set
    existing = set(current_set)
    candidates = [c for c in candidates if c not in existing]
    candidates = list(set(candidates))
    
    # Greedily try to add points
    best_set = current_set[:]
    
    import random
    random.seed(123)
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current = best_set[:]
        
        for p in shuffled:
            if time.time() - start_time > time_limit:
                break
            
            # Check rational distances to all existing
            valid = True
            for q in current:
                if not _is_rational_dist(p, q, D):
                    valid = False
                    break
            if not valid:
                continue
            
            # Check collinearity
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if _collinear(p, current[i], current[j]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check concyclicity
            if len(current) >= 3:
                for i, j, k in combinations(range(len(current)), 3):
                    if _concyclic(p, current[i], current[j], current[k], D):
                        valid = False
                        break
            if not valid:
                continue
            
            current.append(p)
            
            if len(current) > len(best_set):
                best_set = current[:]
                if len(best_set) >= 8:
                    return best_set
        
        if len(best_set) >= 8:
            break
    
    return best_set

def construct():
    """Construct a set of integral point sets in general position."""
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    best_result = None
    best_D = 1
    best_count = 0
    
    # Step 1: Try known 6-point sets
    known_pts, known_D = _try_known_6point_sets()
    if known_pts is not None:
        best_result = known_pts
        best_D = known_D
        best_count = len(known_pts)
    
    # Step 2: If we have a 6-point set, try to extend it
    if best_result is not None and best_count >= 6:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 5:
            extended = _search_extension(best_result, best_D, max_extra=3, time_limit=min(60, remaining_time))
            if len(extended) > best_count:
                best_result = extended
                best_count = len(extended)
    
    # Step 3: If we only have 5 or fewer, try broader search
    if best_count < 6:
        # Known 5-point set as fallback
        points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        frac_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
        if _validate(frac_5, 1):
            if best_result is None or best_count < 5:
                best_result = frac_5
                best_D = 1
                best_count = 5
        
        # Try to extend the 5-point set
        remaining_time = deadline - time.time() - 10
        if remaining_time > 5 and best_result is not None:
            extended = _search_extension(best_result, best_D, max_extra=3, time_limit=min(60, remaining_time))
            if len(extended) > best_count:
                best_result = extended
                best_count = len(extended)
    
    # Step 4: If still nothing good, try direct construction with D=2
    if best_count < 6:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 10:
            # Try D=2 with algebraic approach
            # Points (x, y*sqrt(2)) where x,y rational
            # Distance squared: (x1-x2)^2 + 2*(y1-y2)^2
            # For this to be rational squared, need specific conditions
            
            # Try simple points
            candidates_D2 = []
            for x in range(-20, 21):
                for y in range(-20, 21):
                    candidates_D2.append((Fraction(x), Fraction(y)))
            
            # Try to build a set greedily from D=2 candidates
            current = []
            import random
            random.seed(456)
            shuffled = candidates_D2[:]
            random.shuffle(shuffled)
            
            for p in shuffled:
                if time.time() > deadline - 5:
                    break
                valid = True
                for q in current:
                    if not _is_rational_dist(p, q, 2):
                        valid = False
                        break
                if not valid:
                    continue
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if _collinear(p, current[i], current[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                if len(current) >= 3:
                    for i, j, k in combinations(range(len(current)), 3):
                        if _concyclic(p, current[i], current[j], current[k], 2):
                            valid = False
                            break
                if not valid:
                    continue
                current.append(p)
                if len(current) > best_count:
                    best_result = current[:]
                    best_D = 2
                    best_count = len(current)
                if best_count >= 8:
                    break
    
    # Fallback: ensure we return at least 3 points
    if best_result is None or len(best_result) < 3:
        # Simple 3-point set: equilateral triangle won't work, use right triangle
        best_result = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        best_D = 1
        best_count = 3
    
    # Format the result
    points_formatted = []
    for x, y in best_result:
        if isinstance(x, Fraction):
            x_num, x_den = x.numerator, x.denominator
        else:
            x_num, x_den = int(x), 1
        if isinstance(y, Fraction):
            y_num, y_den = y.numerator, y.denominator
        else:
            y_num, y_den = int(y), 1
        points_formatted.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': best_D, 'points': points_formatted}

# EVOLVE-BLOCK-END
