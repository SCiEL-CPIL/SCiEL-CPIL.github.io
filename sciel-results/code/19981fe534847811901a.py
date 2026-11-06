"""Integral point sets in general position via circle parametrization + Möbius transform."""
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

def _is_squarefree(n):
    n = abs(n)
    if n == 0:
        return False
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _circle_point(t_num, t_den):
    """Rational parametrization of unit circle: ((1-t^2)/(1+t^2), 2t/(1+t^2))."""
    t = Fraction(t_num, t_den)
    t2 = t * t
    denom = 1 + t2
    x = (1 - t2) / denom
    y = 2 * t / denom
    return (x, y)

def _rational_distance_sq(p1, p2):
    """Compute squared distance between two points with rational coords."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _is_perfect_square_frac(frac):
    """Check if a Fraction is a perfect square of a rational number."""
    if frac < 0:
        return False
    num = frac.numerator
    den = frac.denominator
    sqrt_num = isqrt(num)
    sqrt_den = isqrt(den)
    return sqrt_num * sqrt_num == num and sqrt_den * sqrt_den == den

def _check_rational_distances(points):
    """Check all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _rational_distance_sq(points[i], points[j])
            if not _is_perfect_square_frac(d2):
                return False
    return True

def _check_no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    
                    # Four points concyclic iff determinant is zero:
                    # |x1^2+y1^2, x1, y1, 1|
                    # |x2^2+y2^2, x2, y2, 1| = 0
                    # |x3^2+y3^2, x3, y3, 1|
                    # |x4^2+y4^2, x4, y4, 1|
                    
                    def det3(m):
                        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                               - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                               + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
                    
                    def det4(m):
                        return (m[0][0]*det3([m[1][1:], m[2][1:], m[3][1:]])
                               - m[0][1]*det3([m[1][0:1]+m[1][2:], m[2][0:1]+m[2][2:], m[3][0:1]+m[3][2:]])
                               + m[0][2]*det3([m[1][0:2], m[2][0:2], m[3][0:2]])
                               - m[0][3]*det3([m[1][0:3], m[2][0:3], m[3][0:3]]))
                    
                    m = [
                        [x1*x1 + y1*y1, x1, y1, 1],
                        [x2*x2 + y2*y2, x2, y2, 1],
                        [x3*x3 + y3*y3, x3, y3, 1],
                        [x4*x4 + y4*y4, x4, y4, 1]
                    ]
                    if det4(m) == 0:
                        return False
    return True

def _mobius_transform(point, a, b, c):
    """Apply Möbius transformation to break concyclicity while preserving rationality.
    (x, y) -> ((x - a*y)/(1 + b*x), (y + c*x)/(1 + b*x))
    """
    x, y = point
    denom = 1 + b * x
    if denom == 0:
        return None
    x_new = (x - a * y) / denom
    y_new = (y + c * x) / denom
    return (x_new, y_new)

def _try_mobius_set(t_values, a, b, c):
    """Try to build a valid set from given t values and Möbius parameters."""
    transformed = []
    for t in t_values:
        p = _circle_point(t.numerator, t.denominator)
        tp = _mobius_transform(p, a, b, c)
        if tp is None:
            return None
        transformed.append(tp)
    
    # Check distinctness
    seen = set()
    for p in transformed:
        key = (p[0].numerator, p[0].denominator, p[1].numerator, p[1].denominator)
        if key in seen:
            return None
        seen.add(key)
    
    # Check rational distances
    if not _check_rational_distances(transformed):
        return None
    
    # Check no three collinear
    if not _check_no_three_collinear(transformed):
        return None
    
    # Check no four concyclic
    if not _check_no_four_concyclic(transformed):
        return None
    
    return transformed

def _search_mobius():
    """Search for a large set using Möbius transformations of circle points."""
    best_points = None
    best_size = 0
    
    # Generate a pool of t values
    t_pool = []
    for t_num in range(-10, 11):
        for t_den in range(1, 11):
            g = gcd(abs(t_num), t_den)
            if g > 1:
                t_num //= g
                t_den //= g
            t_pool.append(Fraction(t_num, t_den))
    
    # Remove duplicates
    t_pool = list(set(t_pool))
    
    # Try various Möbius parameters
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                # Try to build larger sets by selecting subsets
                for size in range(12, 2, -1):
                    # Try all combinations of t values of given size (limited search)
                    # To keep it fast, we'll try a few random subsets
                    import random
                    random.seed(42)
                    for _ in range(50):
                        subset = random.sample(t_pool, min(size, len(t_pool)))
                        result = _try_mobius_set(subset, a, b, c)
                        if result is not None and len(result) > best_size:
                            best_size = len(result)
                            best_points = result
                            if best_size >= 12:
                                return best_points
    
    return best_points

def _search_integer_grid():
    """Search for a large set in the integer grid with D=1."""
    best_points = None
    best_size = 0
    
    # Generate candidate points in a reasonable grid
    candidates = []
    for x in range(-30, 31):
        for y in range(-30, 31):
            candidates.append((Fraction(x, 1), Fraction(y, 1)))
    
    # Precompute which pairs have rational (i.e., integer) distances
    n = len(candidates)
    valid_pair = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _rational_distance_sq(candidates[i], candidates[j])
            if _is_perfect_square_frac(d2):
                valid_pair[i][j] = True
                valid_pair[j][i] = True
    
    # Greedy construction: try to build a large set
    # Start with different initial points and greedily add
    for start in range(n):
        current = [candidates[start]]
        for candidate in candidates:
            if candidate == current[0]:
                continue
            # Check if this candidate can be added
            can_add = True
            for existing in current:
                idx_c = candidates.index(candidate)
                idx_e = candidates.index(existing)
                if not valid_pair[idx_c][idx_e]:
                    can_add = False
                    break
            if not can_add:
                continue
            
            # Check no three collinear
            if not _check_no_three_collinear(current + [candidate]):
                continue
            
            # Check no four concyclic
            if not _check_no_four_concyclic(current + [candidate]):
                continue
            
            current.append(candidate)
        
        if len(current) > best_size:
            best_size = len(current)
            best_points = current
            if best_size >= 12:
                break
    
    return best_points

def _known_5_point_set():
    """Return a known valid 5-point set."""
    return [
        (Fraction(0, 1), Fraction(0, 1)),
        (Fraction(-56, 1), Fraction(33, 1)),
        (Fraction(-16, 1), Fraction(63, 1)),
        (Fraction(0, 1), Fraction(33, 1)),
        (Fraction(16, 1), Fraction(63, 1))
    ]

def construct():
    """Find a set of integral points in general position with rational distances."""
    # Try integer grid search first (fast for D=1)
    points = _search_integer_grid()
    
    # Try Möbius search if grid search didn't find enough
    if points is None or len(points) < 6:
        mobius_points = _search_mobius()
        if mobius_points is not None and (points is None or len(mobius_points) > len(points)):
            points = mobius_points
    
    # Fallback to known 5-point set
    if points is None or len(points) < 3:
        points = _known_5_point_set()
    
    # Convert to required format
    result_points = []
    for x, y in points:
        result_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
    
    return {'D': 1, 'points': result_points}
