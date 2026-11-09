import time
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

_T0 = time.time()
_DEADLINE = _T0 + 145.0

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
            rn = isqrt(num)
            rd = isqrt(den)
            if rn * rn != num or rd * rd != den:
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

def _try_extend_circle(current, D, max_radius=80):
    """Find a new point with integral distances to all points in current using circle intersection."""
    n = len(current)
    if n < 2:
        return None
    
    x1, y1 = current[0]
    x2, y2 = current[1]
    
    # Distance between points 0 and 1
    dx01 = x1 - x2
    dy01 = y1 - y2
    d01_sq = dx01 * dx01 + D * dy01 * dy01
    num = d01_sq.numerator
    den = d01_sq.denominator
    g = gcd(num, den)
    num //= g
    den //= g
    rn = isqrt(num)
    rd = isqrt(den)
    if rn * rn != num or rd * rd != den:
        return None
    d01 = Fraction(rn, rd)
    
    for r1 in range(1, max_radius + 1):
        if _deadline_hit():
            return None
        for r2 in range(1, max_radius + 1):
            if _deadline_hit():
                return None
            if r1 + r2 < d01 or abs(r1 - r2) > d01:
                continue
            
            A = x2 - x1
            B = D * (y2 - y1)
            C = (Fraction(r1*r1) - Fraction(r2*r2) - x1*x1 + x2*x2 - D*(y1*y1 - y2*y2)) / 2
            
            if A == 0 and B == 0:
                continue
            
            if A != 0:
                a = B*B + D*A*A
                b = -2*B*(C - A*x1) - 2*D*A*A*y1
                c = (C - A*x1)*(C - A*x1) + D*A*A*y1*y1 - Fraction(r1*r1)*A*A
                
                disc = b*b - 4*a*c
                if disc < 0:
                    continue
                
                dnum = disc.numerator
                dden = disc.denominator
                dg = gcd(dnum, dden)
                dnum //= dg
                dden //= dg
                drn = isqrt(dnum)
                drd = isqrt(dden)
                if drn*drn != dnum or drd*drd != dden:
                    continue
                
                sqrt_disc = Fraction(drn, drd)
                
                for sign in [1, -1]:
                    y_val = (-b + sign*sqrt_disc) / (2*a)
                    x_val = (C - B*y_val) / A
                    new_pt = (x_val, y_val)
                    
                    if new_pt in current:
                        continue
                    
                    valid = True
                    for (px, py) in current:
                        dx = x_val - px
                        dy = y_val - py
                        d2 = dx*dx + D*dy*dy
                        num2 = d2.numerator
                        den2 = d2.denominator
                        g2 = gcd(num2, den2)
                        num2 //= g2
                        den2 //= g2
                        rn2 = isqrt(num2)
                        rd2 = isqrt(den2)
                        if rn2*rn2 != num2 or rd2*rd2 != den2:
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
                a = A*A + D*B*B
                b = -2*A*(C - B*y1) - 2*D*B*B*x1
                c = (C - B*y1)*(C - B*y1) + D*B*B*x1*x1 - Fraction(r1*r1)*B*B
                
                disc = b*b - 4*a*c
                if disc < 0:
                    continue
                
                dnum = disc.numerator
                dden = disc.denominator
                dg = gcd(dnum, dden)
                dnum //= dg
                dden //= dg
                drn = isqrt(dnum)
                drd = isqrt(dden)
                if drn*drn != dnum or drd*drd != dden:
                    continue
                
                sqrt_disc = Fraction(drn, drd)
                
                for sign in [1, -1]:
                    x_val = (-b + sign*sqrt_disc) / (2*a)
                    y_val = (C - A*x_val) / B
                    new_pt = (x_val, y_val)
                    
                    if new_pt in current:
                        continue
                    
                    valid = True
                    for (px, py) in current:
                        dx = x_val - px
                        dy = y_val - py
                        d2 = dx*dx + D*dy*dy
                        num2 = d2.numerator
                        den2 = d2.denominator
                        g2 = gcd(num2, den2)
                        num2 //= g2
                        den2 //= g2
                        rn2 = isqrt(num2)
                        rd2 = isqrt(den2)
                        if rn2*rn2 != num2 or rd2*rd2 != den2:
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
    
    return None

def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    best = None
    best_score = -1
    best_D = 1

    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]
    
    if _check_integral(known_5, 1) and _check_general_position(known_5, 1):
        best = list(known_5)
        best_score = 5
        best_D = 1
    
    # Try to extend the known 5-point set
    current = list(known_5)
    max_radius = 50
    while not _deadline_hit() and len(current) < 12:
        new_pt = _try_extend_circle(current, 1, max_radius)
        if new_pt is not None:
            current.append(new_pt)
            if len(current) > best_score:
                best_score = len(current)
                best = list(current)
                best_D = 1
            max_radius += 20
        else:
            break
    
    # Try other D values
    D_candidates = [3, 5, 7, 11, 13]
    for D in D_candidates:
        if _deadline_hit():
            break
        if not _is_squarefree(D):
            continue
        
        start_sets = [
            [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))],
            [(Fraction(0), Fraction(0)), (Fraction(0), Fraction(1))],
            [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))],
        ]
        
        for init_pts in start_sets:
            if _deadline_hit():
                break
            if not _check_integral(init_pts, D):
                continue
            if not _check_general_position(init_pts, D):
                continue
            
            current = list(init_pts)
            max_radius = 30
            while not _deadline_hit() and len(current) < 12:
                new_pt = _try_extend_circle(current, D, max_radius)
                if new_pt is not None:
                    current.append(new_pt)
                    if len(current) > best_score:
                        best_score = len(current)
                        best = list(current)
                        best_D = D
                    max_radius += 10
                else:
                    break
    
    if best is None or best_score < 5:
        return _build_result(known_5, 1)
    
    return _build_result(best, best_D)
