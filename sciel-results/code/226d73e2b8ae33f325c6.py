"""Combined program: algebraic structure + focused grid-search refinement."""
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
    if n % 4 == 0:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n > 1:
        if n % 4 == 0:
            return False
        r = isqrt(n)
        if r * r == n:
            return False
    return True

# --- Fast integer-geometry helpers (for D=1 integer points) ---

def _dist2_int(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _is_int_dist(p1, p2):
    return _is_square(_dist2_int(p1, p2))

def _collinear_int(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _concyclic_int(p1, p2, p3, p4):
    a1, b1 = p1
    a2, b2 = p2
    a3, b3 = p3
    a4, b4 = p4
    if _collinear_int(p1, p2, p3):
        return False
    rhs1 = -(a2 * a2 - a1 * a1 + b2 * b2 - b1 * b1)
    rhs2 = -(a3 * a3 - a1 * a1 + b3 * b3 - b1 * b1)
    x1 = a2 - a1
    y1 = b2 - b1
    x2 = a3 - a1
    y2 = b3 - b1
    det = x1 * y2 - x2 * y1
    if det == 0:
        return False
    num_D = rhs1 * y2 - rhs2 * y1
    num_E = x1 * rhs2 - x2 * rhs1
    num_F = -((a1 * a1 + b1 * b1) * det + num_D * a1 + num_E * b1)
    check = (a4 * a4 + b4 * b4) * det + num_D * a4 + num_E * b4 + num_F
    return check == 0

# --- General (Fraction-based) helpers for D != 1 ---

def _dist_sq_frac(p1, p2, D):
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _is_rational_dist_frac(p1, p2, D):
    d2 = _dist_sq_frac(p1, p2, D)
    return _is_square(d2.numerator) and _is_square(d2.denominator)

def _collinear_frac(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _concyclic_frac(p1, p2, p3, p4, D):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    A = x1 * x1 + D * y1 * y1
    B = x2 * x2 + D * y2 * y2
    C = x3 * x3 + D * y3 * y3
    E = x4 * x4 + D * y4 * y4

    def det3(xa, ya, xb, yb, xc, yc):
        return xa * (yb - yc) - ya * (xb - xc) + (xb * yc - xc * yb)

    det = (A * det3(x2, y2, x3, y3, x4, y4)
           - B * det3(x1, y1, x3, y3, x4, y4)
           + C * det3(x1, y1, x2, y2, x4, y4)
           - E * det3(x1, y1, x2, y2, x3, y3))
    return det == 0

def _validate_int(points):
    """Validate a set of integer points (D=1)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_int_dist(points[i], points[j]):
                return False
    for i, j, k in combinations(range(n), 3):
        if _collinear_int(points[i], points[j], points[k]):
            return False
    for i, j, k, l in combinations(range(n), 4):
        if _concyclic_int(points[i], points[j], points[k], points[l]):
            return False
    return True

def _validate_frac(points, D):
    """Validate a set of Fraction points with given D."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_rational_dist_frac(points[i], points[j], D):
                return False
    for i, j, k in combinations(range(n), 3):
        if _collinear_frac(points[i], points[j], points[k]):
            return False
    for i, j, k, l in combinations(range(n), 4):
        if _concyclic_frac(points[i], points[j], points[k], points[l], D):
            return False
    return True

def _search_6th_int(base_points, deadline, max_limit=600):
    """Search for a 6th integer point extending base_points (D=1)."""
    n = len(base_points)
    base_set = set(base_points)
    # Precompute all pairs and triples for faster checking
    pairs = list(combinations(range(n), 2))
    triples = list(combinations(range(n), 3))
    
    # Try progressively larger search boxes
    for limit in [30, 50, 80, 120, 160, 200, 300, 400, 500, max_limit]:
        for x in range(-limit, limit + 1):
            if time.time() > deadline:
                return None
            for y in range(-limit, limit + 1):
                if time.time() > deadline:
                    return None
                p = (x, y)
                if p in base_set:
                    continue
                # Distances to all base points
                ok = True
                for q in base_points:
                    if not _is_int_dist(p, q):
                        ok = False
                        break
                if not ok:
                    continue
                # Collinearity with any pair
                for i, j in pairs:
                    if _collinear_int(p, base_points[i], base_points[j]):
                        ok = False
                        break
                if not ok:
                    continue
                # Concyclicity with any triple
                for i, j, k in triples:
                    if _concyclic_int(p, base_points[i], base_points[j], base_points[k]):
                        ok = False
                        break
                if not ok:
                    continue
                # Found a valid 6th point
                return base_points + [p]
    return None

def _search_7th_int(base_points, deadline, max_limit=400):
    """Search for a 7th integer point extending base_points (D=1)."""
    n = len(base_points)
    base_set = set(base_points)
    pairs = list(combinations(range(n), 2))
    triples = list(combinations(range(n), 3))
    
    for limit in [40, 60, 100, 150, 200, 300, max_limit]:
        for x in range(-limit, limit + 1):
            if time.time() > deadline:
                return None
            for y in range(-limit, limit + 1):
                if time.time() > deadline:
                    return None
                p = (x, y)
                if p in base_set:
                    continue
                ok = True
                for q in base_points:
                    if not _is_int_dist(p, q):
                        ok = False
                        break
                if not ok:
                    continue
                for i, j in pairs:
                    if _collinear_int(p, base_points[i], base_points[j]):
                        ok = False
                        break
                if not ok:
                    continue
                for i, j, k in triples:
                    if _concyclic_int(p, base_points[i], base_points[j], base_points[k]):
                        ok = False
                        break
                if not ok:
                    continue
                return base_points + [p]
    return None

def _hilbert_parameterization(D, max_t=30, max_r=15):
    """Generate points on circles in Q(sqrt(D)) rationally distant from origin."""
    points = []
    seen = set()
    for r in range(1, max_r + 1):
        for t_num in range(-max_t, max_t + 1):
            for t_den in range(1, max_t + 1):
                if gcd(t_num, t_den) != 1:
                    continue
                t = Fraction(t_num, t_den)
                denom = 1 + D * t * t
                if denom == 0:
                    continue
                x = r * (1 - D * t * t) / denom
                y = r * 2 * t / denom
                if abs(x.numerator) > 10**40 or abs(x.denominator) > 10**40:
                    continue
                if abs(y.numerator) > 10**40 or abs(y.denominator) > 10**40:
                    continue
                key = (x, y)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    return points

def _search_general(D, deadline, max_points=7):
    """Search for a valid set with given D using Hilbert parameterization + greedy."""
    candidates = _hilbert_parameterization(D)
    if not candidates:
        return None

    best_set = None
    import random
    rng = random.Random(42)
    for attempt in range(30):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        current = []
        for p in shuffled:
            if time.time() > deadline:
                break
            ok = True
            for q in current:
                if p == q or not _is_rational_dist_frac(p, q, D):
                    ok = False
                    break
            if not ok:
                continue
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if _collinear_frac(p, current[i], current[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            if len(current) >= 3:
                for i, j, k in combinations(range(len(current)), 3):
                    if _concyclic_frac(p, current[i], current[j], current[k], D):
                        ok = False
                        break
            if not ok:
                continue
            current.append(p)
            if best_set is None or len(current) > len(best_set):
                best_set = current[:]
            if len(best_set) >= max_points:
                return best_set
    return best_set

def _try_base_sets(deadline):
    """Try multiple base sets and search for extensions."""
    # Known valid 5-point sets
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (0, 5), (4, 0), (-3, 4)],
        [(0, 0), (5, 0), (0, 5), (5, 5), (3, 4)],
        [(0, 0), (8, 0), (0, 6), (8, 6), (4, 3)],
    ]
    
    best_result = None
    best_count = 0
    
    for base in base_sets:
        if time.time() > deadline:
            break
        if not _validate_int(base):
            continue
        if len(base) > best_count:
            best_count = len(base)
            best_result = base[:]
        
        found = _search_6th_int(base, deadline)
        if found and len(found) > best_count:
            best_count = len(found)
            best_result = found[:]
    
    return best_result, best_count

def _try_quadratic_6point(deadline):
    """Try to construct a 6-point set in Q(sqrt(D)) for small squarefree D."""
    # For D=3, try hexagon-like configurations
    # Points (x, y) represent (x, y*sqrt(D))
    # In Q(sqrt(3)), rational distances require x1-x2 and y1-y2 such that
    # (dx)^2 + 3*(dy)^2 is a rational square
    
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
        if time.time() > deadline:
            break
        if not _is_squarefree(D):
            continue
        
        # Generate candidate points using rational parameterization
        candidates = []
        for r in range(1, 8):
            for t_num in range(-10, 11):
                for t_den in range(1, 11):
                    if gcd(t_num, t_den) != 1:
                        continue
                    t = Fraction(t_num, t_den)
                    denom = 1 + D * t * t
                    if denom == 0:
                        continue
                    x = r * (1 - D * t * t) / denom
                    y = r * 2 * t / denom
                    if abs(x.numerator) <= 10**40 and abs(x.denominator) <= 10**40:
                        if abs(y.numerator) <= 10**40 and abs(y.denominator) <= 10**40:
                            candidates.append((x, y))
        
        # Also add origin and simple points
        candidates.append((Fraction(0), Fraction(0)))
        for a in range(-5, 6):
            for b in range(-5, 6):
                candidates.append((Fraction(a), Fraction(b)))
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        # Greedy construction
        best = []
        import random
        rng = random.Random(123)
        for attempt in range(20):
            if time.time() > deadline:
                break
            shuffled = unique_candidates[:]
            rng.shuffle(shuffled)
            current = []
            for p in shuffled:
                if time.time() > deadline:
                    break
                ok = True
                for q in current:
                    if p == q or not _is_rational_dist_frac(p, q, D):
                        ok = False
                        break
                if not ok:
                    continue
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        if _collinear_frac(p, current[i], current[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                if len(current) >= 3:
                    for i, j, k in combinations(range(len(current)), 3):
                        if _concyclic_frac(p, current[i], current[j], current[k], D):
                            ok = False
                            break
                if not ok:
                    continue
                current.append(p)
                if len(current) > len(best):
                    best = current[:]
                if len(best) >= 6:
                    return best, D
        
        if len(best) >= 6:
            return best, D
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s safety margin before 120s limit

    # Try multiple base sets for integer points
    best_result, best_count = _try_base_sets(deadline)
    
    # If no valid base set found, use fallback
    if best_result is None:
        base_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        if _validate_int(base_5):
            best_result = base_5
            best_count = 5
        else:
            return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}
    
    # If we have 6, try to find a 7th
    if best_count >= 6 and time.time() < deadline - 15:
        found_7 = _search_7th_int(best_result, deadline)
        if found_7 and len(found_7) > best_count:
            best_result = found_7
            best_count = len(found_7)
    
    # If time remains, try quadratic fields
    best_D = 1
    if time.time() < deadline - 20 and best_count < 7:
        quad_result, quad_D = _try_quadratic_6point(deadline)
        if quad_result and len(quad_result) > best_count:
            best_count = len(quad_result)
            best_result = quad_result
            best_D = quad_D
    
    # Format result
    points_formatted = []
    for p in best_result:
        x, y = p
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
