"""Rewritten program: search for larger integral-point sets (D=1) with rational distances,
no three collinear, no four concyclic, using exact rational arithmetic and a deadline guard."""
import time
from fractions import Fraction
from math import isqrt, gcd

_START = time.time()
_DEADLINE = _START + 130.0  # safety margin before 150s kill

def _sq(x):
    return x * x

def _sqdist(p, q):
    # p, q are (Fraction, Fraction)
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _is_perfect_square_frac(f):
    # f is Fraction >= 0; check if sqrt(f) is rational (i.e. f is a square of a rational)
    if f < 0:
        return False
    if f == 0:
        return True
    n, d = f.numerator, f.denominator
    # n and d must both be perfect squares
    rn = isqrt(n)
    if rn * rn != n:
        return False
    rd = isqrt(d)
    if rd * rd != d:
        return False
    return True

def _collinear(a, b, c):
    # a, b, c are (Fraction, Fraction)
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0

def _concyclic(a, b, c, d):
    # Check if a,b,c,d are concyclic using determinant of 4x4 matrix.
    # Points are (x, y). Circle equation: x^2 + y^2 + Dx + Ey + F = 0
    # Determinant:
    # | x^2+y^2  x  y  1 |
    # | x^2+y^2  x  y  1 |
    # | x^2+y^2  x  y  1 |
    # | x^2+y^2  x  y  1 |
    # We compute this determinant exactly.
    def det4(m):
        # m is 4x4 list of Fractions
        return (
            m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
            - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
            + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
        )
    m = []
    for p in (a, b, c, d):
        m.append([_sq(p[0]) + _sq(p[1]), p[0], p[1], Fraction(1)])
    return det4(m) == 0

def _valid_set(pts):
    # pts: list of (Fraction, Fraction)
    n = len(pts)
    if n < 3:
        return False
    # All pairwise distances rational (D=1, so squared distance must be perfect square of rational)
    for i in range(n):
        for j in range(i + 1, n):
            sd = _sqdist(pts[i], pts[j])
            if not _is_perfect_square_frac(sd):
                return False
    # No three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    # No four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _try_add_point(existing, candidate):
    # Check if adding candidate to existing keeps all constraints
    n = len(existing)
    for i in range(n):
        sd = _sqdist(existing[i], candidate)
        if not _is_perfect_square_frac(sd):
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(existing[i], existing[j], candidate):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic(existing[i], existing[j], existing[k], candidate):
                    return False
    return True

def _search_lattice(max_coord=200, max_points=7, timeout=120.0):
    """Search for a set of points with integer coordinates, all pairwise distances integer,
    no three collinear, no four concyclic."""
    best = None
    # Start with a known 5-point set
    seed = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)), (Fraction(16), Fraction(63))]
    if _valid_set(seed):
        best = list(seed)
    
    # Try to add 6th and 7th points by scanning lattice
    current = list(seed)
    for target_n in range(6, max_points + 1):
        found = False
        for x in range(-max_coord, max_coord + 1):
            if time.time() > timeout:
                break
            for y in range(-max_coord, max_coord + 1):
                if time.time() > timeout:
                    break
                cand = (Fraction(x), Fraction(y))
                if cand in current:
                    continue
                if _try_add_point(current, cand):
                    current.append(cand)
                    found = True
                    if len(current) > (len(best) if best else 0):
                        best = list(current)
                    break
            if found:
                break
        if not found:
            break
    return best

def construct():
    deadline = time.time() + 125.0  # safety margin
    
    # Start with known 5-point set
    seed = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)), (Fraction(16), Fraction(63))]
    best = list(seed)
    
    # Verify seed
    if _valid_set(seed):
        best = list(seed)
    else:
        # Fallback to trivial 3-point set
        best = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
    
    # Search for 6th point
    for x in range(-200, 201):
        if time.time() > deadline:
            break
        for y in range(-200, 201):
            if time.time() > deadline:
                break
            cand = (Fraction(x), Fraction(y))
            if cand in best:
                continue
            if _try_add_point(best, cand):
                best.append(cand)
                break
        else:
            continue
        break
    
    # If we found a 6-point set, try to find a 7th
    if len(best) >= 6:
        for x in range(-200, 201):
            if time.time() > deadline:
                break
            for y in range(-200, 201):
                if time.time() > deadline:
                    break
                cand = (Fraction(x), Fraction(y))
                if cand in best:
                    continue
                if _try_add_point(best, cand):
                    best.append(cand)
                    break
            else:
                continue
            break
    
    # Format result
    D = 1
    points_out = []
    for (x, y) in best:
        x_num = x.numerator
        x_den = x.denominator
        y_num = y.numerator
        y_den = y.denominator
        points_out.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': D, 'points': points_out}
