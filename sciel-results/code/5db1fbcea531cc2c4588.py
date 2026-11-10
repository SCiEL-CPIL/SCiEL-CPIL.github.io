"""Search for rational-distance point sets in general position."""
import time
import math
import random
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _is_squarefree(n):
    if n <= 1:
        return True
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def _sq_dist_sq(p1, p2, D):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _is_rational_square(fr):
    if fr < 0:
        return False
    return _is_square(fr.numerator) and _is_square(fr.denominator)

def _check_all_distances_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_rational_square(_sq_dist_sq(points[i], points[j], D)):
                return False
    return True

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    n = len(points)
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[l]
        x2p, y2p = x2 - x1, y2 - y1
        x3p, y3p = x3 - x1, y3 - y1
        x4p, y4p = x4 - x1, y4 - y1
        q2 = x2p * x2p + D * y2p * y2p
        q3 = x3p * x3p + D * y3p * y3p
        q4 = x4p * x4p + D * y4p * y4p
        det = (x2p * (y3p * (-q4) - y4p * (-q3))
               - y2p * (x3p * (-q4) - x4p * (-q3))
               + (-q2) * (x3p * y4p - x4p * y3p))
        if det == 0:
            return False
    return True

def _validate(points, D):
    if len(points) < 3:
        return False
    if not _is_squarefree(D):
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    if not _check_all_distances_rational(points, D):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def _build_known_5points():
    D = 1
    pts = [(Fraction(0), Fraction(0)),
           (Fraction(-56), Fraction(33)),
           (Fraction(-16), Fraction(63)),
           (Fraction(0), Fraction(33)),
           (Fraction(16), Fraction(63))]
    return pts, D

def _try_add_point(current_points, D, cand):
    if cand in current_points:
        return None
    new_set = current_points + [cand]
    if _validate(new_set, D):
        return new_set
    return None

def _search_expand(start_points, D, deadline, max_extra=7):
    current = list(start_points)
    for step in range(max_extra):
        if time.time() > deadline:
            break
        found = False
        for radius in [50, 100, 200, 400]:
            if time.time() > deadline:
                break
            for seed in range(100):
                if time.time() > deadline:
                    break
                rng = random.Random(1000 + step * 1000 + seed)
                xs = [p[0].numerator for p in current]
                ys = [p[1].numerator for p in current]
                min_x = min(xs) - radius
                max_x = max(xs) + radius
                min_y = min(ys) - radius
                max_y = max(ys) + radius
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    x = rng.randint(min_x, max_x)
                    y = rng.randint(min_y, max_y)
                    cand = (Fraction(x), Fraction(y))
                    new_set = _try_add_point(current, D, cand)
                    if new_set is not None:
                        current = new_set
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return current

def _try_pythagorean_expansion(points, D, deadline):
    n = len(points)
    best = None
    xs = [p[0].numerator for p in points]
    ys = [p[1].numerator for p in points]
    center_x = sum(xs) // n
    center_y = sum(ys) // n
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    radius = max(spread_x, spread_y) + 100
    for dx in range(-radius, radius + 1, 1):
        if time.time() > deadline:
            break
        a = center_x + dx
        for dy in range(-radius, radius + 1, 1):
            if time.time() > deadline:
                break
            b = center_y + dy
            cand = (Fraction(a), Fraction(b))
            if cand in points:
                continue
            ok = True
            for p in points:
                if not _is_rational_square(_sq_dist_sq(cand, p, D)):
                    ok = False
                    break
            if not ok:
                continue
            collinear = False
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    if (cand[0] - x1) * (y2 - y1) == (cand[1] - y1) * (x2 - x1):
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            concyclic = False
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x1p, y1p = x1 - cand[0], y1 - cand[1]
                        x2p, y2p = x2 - cand[0], y2 - cand[1]
                        x3p, y3p = x3 - cand[0], y3 - cand[1]
                        q1 = x1p * x1p + D * y1p * y1p
                        q2 = x2p * x2p + D * y2p * y2p
                        q3 = x3p * x3p + D * y3p * y3p
                        det = (x1p * (y2p * (-q3) - y3p * (-q2))
                               - y1p * (x2p * (-q3) - x3p * (-q2))
                               + (-q1) * (x2p * y3p - x3p * y2p))
                        if det == 0:
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            if concyclic:
                continue
            best = points + [cand]
            break
        if best is not None:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    base_points, D = _build_known_5points()
    if not _validate(base_points, D):
        raise RuntimeError("Base configuration failed validation")
    
    best = list(base_points)
    
    result = _try_pythagorean_expansion(best, D, deadline)
    if result is not None and len(result) > len(best):
        best = result
    
    expanded = _search_expand(best, D, deadline, max_extra=3)
    if len(expanded) > len(best):
        best = expanded
    
    if time.time() < deadline - 10:
        rng = random.Random(999)
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            pts = []
            for _ in range(6):
                x = rng.randint(-200, 200)
                y = rng.randint(-200, 200)
                pts.append((Fraction(x), Fraction(y)))
            if _validate(pts, D) and len(pts) > len(best):
                best = pts
                break
    
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}

# EVOLVE-BLOCK-END
