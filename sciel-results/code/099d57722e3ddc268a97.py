# EVOLVE-BLOCK-START
from fractions import Fraction
import math
import time

_start_time = time.time()
_TIMEOUT = 115  # seconds; safety margin before 120s evaluator kill

def _elapsed():
    return time.time() - _start_time

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = math.isqrt(n)
    return r * r == n

def _check_set(pts):
    """
    pts: list of (Fraction x, Fraction y) representing (x, y*sqrt(D)) for D=1.
    Returns (n, ok).
    """
    n = len(pts)
    # distinct
    seen = set()
    for p in pts:
        if p in seen:
            return n, False
        seen.add(p)
    # rational distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            if not _is_square((dx * dx + dy * dy).numerator):
                return n, False
    # no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return n, False
    # no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    # determinant of circle equation
                    A = 2 * (x2 - x1)
                    B = 2 * (y2 - y1)
                    C = x1*x1 + y1*y1 - x2*x2 - y2*y2
                    D = 2 * (x3 - x1)
                    E = 2 * (y3 - y1)
                    F = x1*x1 + y1*y1 - x3*x3 - y3*y3
                    # circle through 1,2,3
                    det = A*E - D*B
                    if det == 0:
                        # collinear, already checked
                        return n, False
                    a = (E*C - B*F) / det
                    b = (A*F - D*C) / det
                    # check if 4 is on circle: (x4-a)^2 + (y4-b)^2 == r^2
                    # r^2 = a^2 + b^2 - (x1^2+y1^2)
                    lhs = (x4 - a)*(x4 - a) + (y4 - b)*(y4 - b)
                    rhs = a*a + b*b - (x1*x1 + y1*y1)
                    if lhs == rhs:
                        return n, False
    return n, True

def _search_add_point(existing, max_iter=20000):
    """Try to find a point to add to existing list."""
    # existing: list of (Fraction, Fraction)
    # try small rationals
    for num in range(-10, 11):
        for den in [1, 2, 3, 4, 5, 6]:
            for num2 in range(-10, 11):
                for den2 in [1, 2, 3, 4, 5, 6]:
                    if _elapsed() > _TIMEOUT:
                        return None
                    cand = (Fraction(num, den), Fraction(num2, den2))
                    # check distinct
                    if cand in existing:
                        continue
                    # check distances to all existing
                    ok = True
                    for p in existing:
                        dx = cand[0] - p[0]
                        dy = cand[1] - p[1]
                        if not _is_square((dx*dx + dy*dy).numerator):
                            ok = False
                            break
                    if not ok:
                        continue
                    # check collinearity with any pair
                    ok = True
                    for i in range(len(existing)):
                        for j in range(i+1, len(existing)):
                            x1, y1 = existing[i]
                            x2, y2 = existing[j]
                            x3, y3 = cand
                            if (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # check concyclic with any triple
                    ok = True
                    for i in range(len(existing)):
                        for j in range(i+1, len(existing)):
                            for k in range(j+1, len(existing)):
                                x1, y1 = existing[i]
                                x2, y2 = existing[j]
                                x3, y3 = existing[k]
                                x4, y4 = cand
                                A = 2*(x2-x1)
                                B = 2*(y2-y1)
                                C = x1*x1+y1*y1 - x2*x2-y2*y2
                                D = 2*(x3-x1)
                                E = 2*(y3-y1)
                                F = x1*x1+y1*y1 - x3*x3-y3*y3
                                det = A*E - D*B
                                if det == 0:
                                    ok = False
                                    break
                                a = (E*C - B*F) / det
                                b = (A*F - D*C) / det
                                lhs = (x4-a)*(x4-a) + (y4-b)*(y4-b)
                                rhs = a*a + b*b - (x1*x1+y1*y1)
                                if lhs == rhs:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        return cand
    return None

def construct():
    _start_time = time.time()
    # Start with the known 5-point set
    base = [(Fraction(0), Fraction(0)),
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]
    
    best = base
    # Try to add a 6th point
    if _elapsed() < _TIMEOUT - 5:
        p6 = _search_add_point(base, max_iter=50000)
        if p6 is not None:
            candidate = base + [p6]
            n, ok = _check_set(candidate)
            if ok:
                best = candidate
    
    # Try to find a better 6-point set by random search
    if _elapsed() < _TIMEOUT - 10:
        import random
        random.seed(42)
        # Try to find 6 points from scratch with small coordinates
        for _ in range(5000):
            if _elapsed() > _TIMEOUT - 10:
                break
            pts = []
            for _ in range(6):
                x = Fraction(random.randint(-20, 20), random.choice([1,2,3,4]))
                y = Fraction(random.randint(-20, 20), random.choice([1,2,3,4]))
                pts.append((x, y))
            n, ok = _check_set(pts)
            if ok and n == 6:
                best = pts
                break
    
    # Format output
    result = {'D': 1, 'points': []}
    for x, y in best:
        result['points'].append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return result
# EVOLVE-BLOCK-END
