import time
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

_T0 = time.time()
_DEADLINE = _T0 + 142.0

def _elapsed():
    return time.time() - _T0

def _deadline_hit():
    return _elapsed() > _DEADLINE

def _is_squarefree(n):
    if n <= 0:
        return False
    if n == 1:
        return True
    for p in range(2, isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_integral(points, D):
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + D * dy * dy
            if d2 < 0:
                return False
            num = d2.numerator
            den = d2.denominator
            g = gcd(num, den)
            num //= g
            den //= g
            if not _is_perfect_square(num) or not _is_perfect_square(den):
                return False
    return True

def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _four_concyclic(p1, p2, p3, p4, D):
    M = []
    for (x, y) in (p1, p2, p3, p4):
        M.append([x * x + D * y * y, x, y, Fraction(1)])
    n = 4
    A = [row[:] for row in M]
    for c in range(n):
        pivot = None
        for r in range(c, n):
            if A[r][c] != 0:
                pivot = r
                break
        if pivot is None:
            return True
        if pivot != c:
            A[c], A[pivot] = A[pivot], A[c]
        for r in range(c + 1, n):
            factor = A[r][c] / A[c][c]
            for cc in range(c, n):
                A[r][cc] -= factor * A[c][cc]
    det = Fraction(1)
    for i in range(n):
        det *= A[i][i]
    return det == 0

def _check_general_position(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _build_result(points, D):
    out = []
    for (x, y) in points:
        out.append([ [x.numerator, x.denominator], [y.numerator, y.denominator] ])
    return {'D': D, 'points': out}

def _dist_sq(p1, p2, D):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _is_rational_dist_sq(d2):
    if d2 < 0:
        return False
    num = d2.numerator
    den = d2.denominator
    g = gcd(num, den)
    num //= g
    den //= g
    return _is_perfect_square(num) and _is_perfect_square(den)

def _try_extend(current, D, max_radius):
    """Find a new point with integral distances to all points in current."""
    n = len(current)
    if n < 2:
        return None
    
    x1, y1 = current[0]
    x2, y2 = current[1]
    
    dx01 = x1 - x2
    dy01 = y1 - y2
    d01_sq = dx01 * dx01 + D * dy01 * dy01
    if not _is_rational_dist_sq(d01_sq):
        return None
    
    num = d01_sq.numerator
    den = d01_sq.denominator
    g = gcd(num, den)
    num //= g
    den //= g
    d01 = Fraction(isqrt(num), isqrt(den))
    
    # Precompute distances from point 0 to all other points
    # and from point 1 to all other points
    for r1 in range(1, max_radius + 1):
        if _deadline_hit():
            return None
        for r2 in range(1, max_radius + 1):
            if _deadline_hit():
                return None
            if r1 + r2 < d01 or abs(r1 - r2) > d01:
                continue
            
            # Solve for intersection of two circles
            # Circle 1: center (x1,y1), radius r1
            # Circle 2: center (x2,y2), radius r2
            # x^2 + D*y^2 = r1^2 + 2*x1*x + 2*D*y1*y - x1^2 - D*y1^2  (circle 1)
            # x^2 + D*y^2 = r2^2 + 2*x2*x + 2*D*y2*y - x2^2 - D*y2^2  (circle 2)
            # Subtract: 2*(x2-x1)*x + 2*D*(y2-y1)*y = r1^2 - r2^2 + x2^2 - x1^2 + D*(y2^2 - y1^2)
            
            A = x2 - x1
            B = D * (y2 - y1)
            C = (Fraction(r1*r1) - Fraction(r2*r2) - x1*x1 + x2*x2 - D*(y1*y1 - y2*y2)) / 2
            
            if A == 0 and B == 0:
                continue
            
            if A != 0:
                # y = (C - A*x) / B if B != 0, else use x directly
                # Substitute into circle 1: x^2 + D*y^2 = r1^2 + 2*x1*x + 2*D*y1*y - x1^2 - D*y1^2
                if B != 0:
                    # y = (C - A*x) / B
                    # x^2 + D*(C-A*x)^2/B^2 = r1^2 + 2*x1*x + 2*D*y1*(C-A*x)/B - x1^2 - D*y1^2
                    # Multiply by B^2:
                    # B^2*x^2 + D*(C-A*x)^2 = r1^2*B^2 + 2*x1*x*B^2 + 2*D*y1*(C-A*x)*B - x1^2*B^2 - D*y1^2*B^2
                    # B^2*x^2 + D*(C^2 - 2*A*C*x + A^2*x^2) = r1^2*B^2 + 2*x1*B^2*x + 2*D*y1*C*B - 2*D*y1*A*B*x - x1^2*B^2 - D*y1^2*B^2
                    # (B^2 + D*A^2)*x^2 + (-2*D*A*C - 2*x1*B^2 + 2*D*y1*A*B)*x + (D*C^2 - r1^2*B^2 - 2*D*y1*C*B + x1^2*B^2 + D*y1^2*B^2) = 0
                    
                    a = B*B + D*A*A
                    b = -2*D*A*C - 2*x1*B*B + 2*D*y1*A*B
                    c = D*C*C - Fraction(r1*r1)*B*B - 2*D*y1*C*B + x1*x1*B*B + D*y1*y1*B*B
                    
                    disc = b*b - 4*a*c
                    if disc < 0:
                        continue
                    
                    dnum = disc.numerator
                    dden = disc.denominator
                    dg = gcd(dnum, dden)
                    dnum //= dg
                    dden //= dg
                    if not _is_perfect_square(dnum) or not _is_perfect_square(dden):
                        continue
                    
                    sqrt_disc = Fraction(isqrt(dnum), isqrt(dden))
                    
                    for sign in [1, -1]:
                        x_val = (-b + sign*sqrt_disc) / (2*a)
                        y_val = (C - A*x_val) / B
                        new_pt = (x_val, y_val)
                        
                        if new_pt in current:
                            continue
                        
                        # Check all distances are rational
                        valid = True
                        for (px, py) in current:
                            if not _is_rational_dist_sq(_dist_sq(new_pt, (px, py), D)):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Check no three collinear
                        collinear_ok = True
                        for i in range(n):
                            for j in range(i+1, n):
                                if _collinear(current[i], current[j], new_pt):
                                    collinear_ok = False
                                    break
                            if not collinear_ok:
                                break
                        if not collinear_ok:
                            continue
                        
                        # Check no four concyclic
                        concyclic_ok = True
                        for i in range(n):
                            for j in range(i+1, n):
                                for k in range(j+1, n):
                                    if _four_concyclic(current[i], current[j], current[k], new_pt, D):
                                        concyclic_ok = False
                                        break
                                if not concyclic_ok:
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            continue
                        
                        return new_pt
                else:
                    # B == 0, so y2 == y1 (since D != 0)
                    # A*x = C, so x = C/A
                    x_val = C / A
                    # Now find y from circle 1: x_val^2 + D*y^2 = r1^2 + 2*x1*x_val + 2*D*y1*y - x1^2 - D*y1^2
                    # D*y^2 - 2*D*y1*y + (x_val^2 - r1^2 - 2*x1*x_val + x1^2 + D*y1^2) = 0
                    a = D
                    b = -2*D*y1
                    c = x_val*x_val - Fraction(r1*r1) - 2*x1*x_val + x1*x1 + D*y1*y1
                    
                    disc = b*b - 4*a*c
                    if disc < 0:
                        continue
                    
                    dnum = disc.numerator
                    dden = disc.denominator
                    dg = gcd(dnum, dden)
                    dnum //= dg
                    dden //= dg
                    if not _is_perfect_square(dnum) or not _is_perfect_square(dden):
                        continue
                    
                    sqrt_disc = Fraction(isqrt(dnum), isqrt(dden))
                    
                    for sign in [1, -1]:
                        y_val = (-b + sign*sqrt_disc) / (2*a)
                        new_pt = (x_val, y_val)
                        
                        if new_pt in current:
                            continue
                        
                        valid = True
                        for (px, py) in current:
                            if not _is_rational_dist_sq(_dist_sq(new_pt, (px, py), D)):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        collinear_ok = True
                        for i in range(n):
                            for j in range(i+1, n):
                                if _collinear(current[i], current[j], new_pt):
                                    collinear_ok = False
                                    break
                            if not collinear_ok:
                                break
                        if not collinear_ok:
                            continue
                        
                        concyclic_ok = True
                        for i in range(n):
                            for j in range(i+1, n):
                                for k in range(j+1, n):
                                    if _four_concyclic(current[i], current[j], current[k], new_pt, D):
                                        concyclic_ok = False
                                        break
                                if not concyclic_ok:
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            continue
                        
                        return new_pt
            
            elif B != 0:
                # A == 0, so x1 == x2
                # B*y = C, so y = C/B
                y_val = C / B
                # x^2 + D*y_val^2 = r1^2 + 2*x1*x + 2*D*y1*y_val - x1^2 - D*y1^2
                # x^2 - 2*x1*x + (D*y_val^2 - r1^2 - 2*D*y1*y_val + x1^2 + D*y1^2) = 0
                a = Fraction(1)
                b = -2*x1
                c = D*y_val*y_val - Fraction(r1*r1) - 2*D*y1*y_val + x1*x1 + D*y1*y1
                
                disc = b*b - 4*a*c
                if disc < 0:
                    continue
                
                dnum = disc.numerator
                dden = disc.denominator
                dg = gcd(dnum, dden)
                dnum //= dg
                dden //= dg
                if not _is_perfect_square(dnum) or not _is_perfect_square(dden):
                    continue
                
                sqrt_disc = Fraction(isqrt(dnum), isqrt(dden))
                
                for sign in [1, -1]:
                    x_val = (-b + sign*sqrt_disc) / (2*a)
                    new_pt = (x_val, y_val)
                    
                    if new_pt in current:
                        continue
                    
                    valid = True
                    for (px, py) in current:
                        if not _is_rational_dist_sq(_dist_sq(new_pt, (px, py), D)):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    collinear_ok = True
                    for i in range(n):
                        for j in range(i+1, n):
                            if _collinear(current[i], current[j], new_pt):
                                collinear_ok = False
                                break
                        if not collinear_ok:
                            break
                    if not collinear_ok:
                        continue
                    
                    concyclic_ok = True
                    for i in range(n):
                        for j in range(i+1, n):
                            for k in range(j+1, n):
                                if _four_concyclic(current[i], points[j], points[k], new_pt, D):
                                    concyclic_ok = False
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        continue
                    
                    return new_pt
    
    return None

def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 142.0

    # Known 5-point integral set (verified)
    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]
    
    best = list(known_5)
    best_D = 1
    best_score = 5
    
    # Verify the known set
    if not (_check_integral(known_5, 1) and _check_general_position(known_5, 1)):
        # Fallback: simple 3-point set
        simple_3 = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        return _build_result(simple_3, 1)
    
    # Try to extend the 5-point set with larger radius
    current = list(known_5)
    for max_r in [30, 50, 80, 120, 200]:
        if _deadline_hit():
            break
        new_pt = _try_extend(current, 1, max_r)
        if new_pt is not None:
            current.append(new_pt)
            if len(current) > best_score:
                best_score = len(current)
                best = list(current)
                best_D = 1
            # Try to extend further
            while not _deadline_hit() and len(current) < 12:
                new_pt2 = _try_extend(current, 1, max_r)
                if new_pt2 is not None:
                    current.append(new_pt2)
                    if len(current) > best_score:
                        best_score = len(current)
                        best = list(current)
                        best_D = 1
                else:
                    break
        else:
            # Try a different pair of anchor points
            # Reorder: try different pairs as anchors
            for i in range(len(current) - 1):
                if _deadline_hit():
                    break
                # Swap to try different anchor pair
                saved = current[:]
                # Try using points i and i+1 as anchors
                # This is already done implicitly; just break
                break
            break
    
    # Try different initial configurations with D=1
    # Simple 2-point starts with different separations
    if not _deadline_hit():
        for sep in [1, 2, 3, 5]:
            if _deadline_hit():
                break
            init = [(Fraction(0), Fraction(0)), (Fraction(sep), Fraction(0))]
            if not _check_integral(init, 1) or not _check_general_position(init, 1):
                continue
            cur = list(init)
            for max_r in [20, 40, 60]:
                if _deadline_hit():
                    break
                new_pt = _try_extend(cur, 1, max_r)
                if new_pt is not None:
                    cur.append(new_pt)
                    while not _deadline_hit() and len(cur) < 12:
                        new_pt2 = _try_extend(cur, 1, max_r)
                        if new_pt2 is not None:
                            cur.append(new_pt2)
                        else:
                            break
                    if len(cur) > best_score:
                        best_score = len(cur)
                        best = list(cur)
                        best_D = 1
                else:
                    break
    
    # Try D=3 with simple starts
    if not _deadline_hit():
        for init in [
            [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))],
            [(Fraction(0), Fraction(0)), (Fraction(0), Fraction(1))],
        ]:
            if _deadline_hit():
                break
            if not _check_integral(init, 3) or not _check_general_position(init, 3):
                continue
            cur = list(init)
            for max_r in [20, 40]:
                if _deadline_hit():
                    break
                new_pt = _try_extend(cur, 3, max_r)
                if new_pt is not None:
                    cur.append(new_pt)
                    while not _deadline_hit() and len(cur) < 12:
                        new_pt2 = _try_extend(cur, 3, max_r)
                        if new_pt2 is not None:
                            cur.append(new_pt2)
                        else:
                            break
                    if len(cur) > best_score:
                        best_score = len(cur)
                        best = list(cur)
                        best_D = 3
                else:
                    break
    
    if best_score < 5:
        return _build_result(known_5, 1)
    
    return _build_result(best, best_D)
