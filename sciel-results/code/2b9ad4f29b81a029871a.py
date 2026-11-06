"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _no_four_concyclic_fast(points, D):
    """Check no four points are concyclic using determinant test.
    Points are (x, y_coeff) where actual y = y_coeff * sqrt(D).
    """
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1c = points[i]
                    x2, y2c = points[j]
                    x3, y3c = points[k]
                    x4, y4c = points[l]
                    
                    # For general D, y_actual = y_coeff * sqrt(D)
                    # det = |x, y_c*sqrt(D), x^2 + D*y_c^2, 1|
                    # Each 3x3 minor = sqrt(D) * bracket
                    def minor3(i1, i2, i3):
                        a, b = points[i1]
                        c, d = points[i2]
                        e, f = points[i3]
                        va = a*a + D*b*b
                        vc = c*c + D*d*d
                        ve = e*e + D*f*f
                        bracket = a*(d*ve - vc*f) - b*(c*ve - vc*e) + va*(c*f - d*e)
                        return bracket
                    
                    m1 = minor3(i, j, k)
                    m2 = minor3(i, j, l)
                    m3 = minor3(i, k, l)
                    m4 = minor3(j, k, l)
                    
                    total = m1 - m2 + m3 - m4
                    if total == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Known 5-point integral point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    base_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    
    best = base_frac
    best_count = 5
    
    # Strategy: For each candidate (x, y), check if distance to ALL base points is integer.
    # Use Pythagorean triple enumeration to generate candidates efficiently.
    
    # For point (a, b), (x-a)^2 + (y-b)^2 = k^2 for some integer k.
    # So we need (x, y) such that for each base point (a_i, b_i),
    # (x - a_i)^2 + (y - b_i)^2 is a perfect square.
    
    # Efficient approach: iterate over x, and for each base point, the condition
    # (x - a_i)^2 + (y - b_i)^2 = k^2 means (y - b_i)^2 = k^2 - (x-a_i)^2.
    # For a fixed x, we can enumerate possible y values that make distance to point 0 integer,
    # then check the rest.
    
    # Even better: for fixed x, distance to (0,0) is integer means x^2 + y^2 = k^2.
    # So y must satisfy y^2 = k^2 - x^2 for some k >= |x|.
    # This gives y = ±sqrt(k^2 - x^2), which is integer only when k^2 - x^2 is a perfect square.
    # i.e., (k, x, y) forms a Pythagorean triple (or y=0).
    
    # Let's enumerate: for each x in range, find all y such that x^2 + y^2 is a perfect square.
    # Then filter by other base points.
    
    search_range = 2000
    
    # Precompute: for each x, the set of y values where x^2+y^2 is a perfect square
    # (distance to (0,0) is integer)
    # Actually, let's just iterate and check all 5 points.
    
    found_six = False
    # Try x from 0 to search_range (by symmetry, x and -x give mirror images)
    # The 5-point set is symmetric about x=0, so if (x,y) works, (-x,y) also works.
    # We only need to check x >= 0 and then we get the mirror for free.
    
    for x in range(0, search_range + 1):
        if time.time() > deadline:
            break
        
        # For this x, find all y such that distance to (0,0) is integer
        # x^2 + y^2 = k^2 => y^2 = k^2 - x^2
        # k ranges from |x| to some max. y^2 = k^2 - x^2 must be a perfect square.
        # Let y^2 = m^2, so k^2 - m^2 = x^2, (k-m)(k+m) = x^2.
        # This is equivalent to finding Pythagorean triples with leg x.
        
        # Simpler: just iterate y and check. But that's slow.
        # Better: for each divisor pair of x^2, get k and m.
        
        # Actually, let's just check all y in a reasonable range for each x.
        # The constraint from point (0,0): x^2 + y^2 must be a perfect square.
        # Let's find all such y.
        
        x2 = x * x
        # y^2 = k^2 - x^2, k >= x
        # Max y: let's say up to search_range
        # For each y, check if x^2 + y^2 is a perfect square.
        
        # To be efficient, let's enumerate k and derive y.
        # k^2 - y^2 = x^2 => (k-y)(k+y) = x^2
        # Let a = k-y, b = k+y, a*b = x^2, a <= b, same parity.
        # k = (a+b)/2, y = (b-a)/2
        
        candidates_y = set()
        if x == 0:
            # y^2 = k^2, so y = ±k for any k. All integers work.
            # But we need to limit range.
            for y in range(-search_range, search_range + 1):
                candidates_y.add(y)
        else:
            x2_val = x * x
            # Find all factor pairs (a, b) with a*b = x2_val, a <= b, a and b same parity
            a = 1
            while a * a <= x2_val:
                if x2_val % a == 0:
                    b = x2_val // a
                    if (a + b) % 2 == 0:  # same parity
                        k_val = (a + b) // 2
                        y_val = (b - a) // 2
                        if y_val >= 0:
                            candidates_y.add(y_val)
                            if y_val != 0:
                                candidates_y.add(-y_val)
                        else:
                            # y_val negative, but we take abs and add both signs
                            y_abs = -y_val
                            candidates_y.add(y_abs)
                            if y_abs != 0:
                                candidates_y.add(-y_abs)
                a += 1
        
        # Now filter candidates_y by checking distance to all other base points
        for y in sorted(candidates_y):
            if time.time() > deadline:
                break
            
            # Check not collinear with any pair of base points
            collinear = False
            for i in range(5):
                for j in range(i+1, 5):
                    x1, y1 = base_frac[i]
                    x2_val, y2_val = base_frac[j]
                    cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                    if cp == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check all pairwise distances are integers
            all_ok = True
            for pt in base_frac:
                dx = x - pt[0]
                dy = y - pt[1]
                d2 = dx * dx + dy * dy
                if not _is_square(d2.numerator * d2.denominator):
                    all_ok = False
                    break
            if not all_ok:
                continue
            
            # Check no four concyclic
            candidate = base_frac + [(Fraction(x), Fraction(y))]
            if not _no_four_concyclic_fast(candidate, D):
                continue
            
            # Found a 6-point set!
            best = candidate
            best_count = 6
            found_six = True
            break
        if found_six:
            break
    
    # If we found 6 points, try to find a 7th
    if found_six:
        found_seven = False
        search_range2 = 3000
        for x in range(0, search_range2 + 1):
            if time.time() > deadline:
                break
            
            x2 = x * x
            candidates_y = set()
            if x == 0:
                for y in range(-search_range2, search_range2 + 1):
                    candidates_y.add(y)
            else:
                x2_val = x * x
                a = 1
                while a * a <= x2_val:
                    if x2_val % a == 0:
                        b = x2_val // a
                        if (a + b) % 2 == 0:
                            k_val = (a + b) // 2
                            y_val = (b - a) // 2
                            if y_val >= 0:
                                candidates_y.add(y_val)
                                if y_val != 0:
                                    candidates_y.add(-y_val)
                            else:
                                y_abs = -y_val
                                candidates_y.add(y_abs)
                                if y_abs != 0:
                                    candidates_y.add(-y_abs)
                    a += 1
            
            for y in sorted(candidates_y):
                if time.time() > deadline:
                    break
                
                # Check not collinear with any pair
                collinear = False
                for i in range(6):
                    for j in range(i+1, 6):
                        x1, y1 = best[i]
                        x2_val, y2_val = best[j]
                        cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                        if cp == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Check all pairwise distances are integers
                all_ok = True
                for pt in best:
                    dx = x - pt[0]
                    dy = y - pt[1]
                    d2 = dx * dx + dy * dy
                    if not _is_square(d2.numerator * d2.denominator):
                        all_ok = False
                        break
                if not all_ok:
                    continue
                
                # Check no four concyclic
                candidate = best + [(Fraction(x), Fraction(y))]
                if not _no_four_concyclic_fast(candidate, D):
                    continue
                
                best = candidate
                best_count = 7
                found_seven = True
                break
            if found_seven:
                break
    
    # If we still only have 5 points, try a wider search with a different approach
    if best_count == 5:
        # Try a broader direct search but with early termination
        search_range3 = 500
        for x in range(0, search_range3 + 1):
            if time.time() > deadline:
                break
            for y in range(0, search_range3 + 1):
                if time.time() > deadline:
                    break
                
                # Check distance to (0,0) first (cheapest)
                d2_00 = x*x + y*y
                if not _is_square(d2_00):
                    continue
                
                # Check distance to (-56, 33)
                dx1 = x + 56
                dy1 = y - 33
                d2_1 = dx1*dx1 + dy1*dy1
                if not _is_square(d2_1):
                    continue
                
                # Check distance to (-16, 63)
                dx2 = x + 16
                dy2 = y - 63
                d2_2 = dx2*dx2 + dy2*dy2
                if not _is_square(d2_2):
                    continue
                
                # Check distance to (0, 33)
                dx3 = x
                dy3 = y - 33
                d2_3 = dx3*dx3 + dy3*dy3
                if not _is_square(d2_3):
                    continue
                
                # Check distance to (16, 63)
                dx4 = x - 16
                dy4 = y - 63
                d2_4 = dx4*dx4 + dy4*dy4
                if not _is_square(d2_4):
                    continue
                
                # All distances integer, now check collinearity
                collinear = False
                for i in range(5):
                    for j in range(i+1, 5):
                        x1, y1 = base_frac[i]
                        x2_val, y2_val = base_frac[j]
                        cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                        if cp == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Check no four concyclic
                candidate = base_frac + [(Fraction(x), Fraction(y))]
                if not _no_four_concyclic_fast(candidate, D):
                    continue
                
                best = candidate
                best_count = 6
                break
            if best_count == 6:
                break
    
    # Format output
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END
