"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import isqrt, gcd
from fractions import Fraction
import time
import random

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _check_point_against_set(x_new, y_new, pts, D):
    """
    Check if adding point (x_new, y_new) to pts maintains all conditions.
    pts: list of (Fraction x, Fraction y)
    Returns True if the new point is compatible.
    """
    n = len(pts)
    if n < 2:
        return True

    # Check distance to all existing points
    for i in range(n):
        dx = x_new - pts[i][0]
        dy = y_new - pts[i][1]
        sq_dist = dx * dx + D * dy * dy
        if sq_dist < 0:
            return False
        num = sq_dist.numerator
        den = sq_dist.denominator
        if not _is_square(num) or not _is_square(den):
            return False

    # Check no three collinear with new point
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            det = (x2 - x1) * (y_new - y1) - (y2 - y1) * (x_new - x1)
            if det == 0:
                return False

    # Check no four concyclic with new point
    if n >= 3:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]

                    M = [
                        [x1 * x1 + y1 * y1 * D, x1, y1, 1],
                        [x2 * x2 + y2 * y2 * D, x2, y2, 1],
                        [x3 * x3 + y3 * y3 * D, x3, y3, 1],
                        [x_new * x_new + y_new * y_new * D, x_new, y_new, 1]
                    ]
                    det = _det4(M)
                    if det == 0:
                        return False

    return True


def _check_all_conditions(points, D):
    """
    Check all conditions for a set of points with given D.
    points: list of (x_num, x_den, y_num, y_den) representing (x, y*sqrt(D))
    Returns True if all conditions are satisfied.
    """
    n = len(points)
    if n < 3:
        return False

    # Convert to Fractions
    pts = []
    for (xn, xd, yn, yd) in points:
        x = Fraction(xn, xd)
        y = Fraction(yn, yd)
        pts.append((x, y))

    # Check all pairwise squared distances are rational
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            sq_dist = dx * dx + D * dy * dy
            if sq_dist < 0:
                return False
            num = sq_dist.numerator
            den = sq_dist.denominator
            if not _is_square(num) or not _is_square(den):
                return False

    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if det == 0:
                    return False

    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]

                    M = []
                    for (x, y) in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
                        M.append([x * x + y * y * D, x, y, 1])

                    det = _det4(M)
                    if det == 0:
                        return False

    return True


def _det4(M):
    """Compute determinant of a 4x4 matrix of Fractions using Gaussian elimination."""
    M = [row[:] for row in M]
    n = 4
    det = Fraction(1, 1)
    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            return Fraction(0, 1)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            det = -det
        det *= M[col][col]
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for k in range(col, n):
                    M[row][k] -= factor * M[col][k]
    return det


def _greedy_grow(pts, candidates, D, time_limit, start_time):
    """
    Greedily add candidates to pts, checking only the new point against existing.
    Returns the grown set.
    """
    current = list(pts)
    current_set = set(current)
    
    # Precompute Fraction versions for speed
    frac_pts = [(Fraction(x), Fraction(y)) for (x, y) in current]
    
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        if cand in current_set:
            continue
        
        x_new = Fraction(cand[0])
        y_new = Fraction(cand[1])
        
        if _check_point_against_set(x_new, y_new, frac_pts, D):
            current.append(cand)
            current_set.add(cand)
            frac_pts.append((x_new, y_new))
    
    return current


def _search_integer_points(max_coord=400, max_points=8, time_limit=100):
    """
    Search for sets of integer points with rational distances, no three collinear, no four concyclic.
    Start from known 5-point sets and try to add more using incremental checks.
    """
    start_time = time.time()

    # Start with the known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    best = list(base_points)

    # Generate candidate points in a grid
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in base_points:
                candidates.append((x, y))

    # Sort candidates by "interestingness" - prefer those with more Pythagorean connections
    # For now, just shuffle
    random.seed(42)
    random.shuffle(candidates)

    # Greedy grow from base
    current = _greedy_grow(base_points, candidates, 1, time_limit, start_time)
    if len(current) > len(best):
        best = list(current)

    # Try different starting sets with random 4-point seeds
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        random.seed(attempt + 100)
        start_pts = []
        used = set()
        for _ in range(4):
            while True:
                x = random.randint(-max_coord, max_coord)
                y = random.randint(-max_coord, max_coord)
                if (x, y) not in used:
                    used.add((x, y))
                    start_pts.append((x, y))
                    break

        # Verify the 4-point set is valid
        if not _check_all_conditions([(x, 1, y, 1) for (x, y) in start_pts], 1):
            continue

        # Greedy grow
        current = _greedy_grow(start_pts, candidates, 1, time_limit, start_time)
        if len(current) > len(best):
            best = list(current)
            if len(best) >= max_points:
                break

    return best


def _generate_pythagorean_points(max_r=200):
    """
    Generate integer points that lie on circles of radius r centered at origin.
    These are more likely to have rational distances to each other.
    """
    points = set()
    for r in range(1, max_r + 1):
        r2 = r * r
        for x in range(-r, r + 1):
            y2 = r2 - x * x
            if y2 < 0:
                continue
            y = isqrt(y2)
            if y * y == y2:
                points.add((x, y))
                if y != 0:
                    points.add((x, -y))
                if x != 0:
                    points.add((-x, y))
                    if y != 0:
                        points.add((-x, -y))
    return list(points)


def _search_with_parametrization(time_limit=50):
    """
    Use Pythagorean points and random points to generate candidates,
    then apply greedy growth with incremental checks.
    """
    start_time = time.time()
    
    # Generate candidate points from Pythagorean triples
    candidates = _generate_pythagorean_points(max_r=150)
    
    # Also include some random integer points
    random.seed(123)
    for _ in range(1000):
        x = random.randint(-200, 200)
        y = random.randint(-200, 200)
        candidates.append((x, y))
    
    # Remove duplicates
    candidates = list(set(candidates))
    random.shuffle(candidates)
    
    best = []
    best_size = 0
    
    # Try multiple random initializations
    for attempt in range(80):
        if time.time() - start_time > time_limit:
            break
        
        # Start with 3-4 random valid points
        random.seed(attempt + 200)
        current = []
        used = set()
        
        # Find initial points that form a valid small set
        for cand in candidates:
            if len(current) >= 4:
                break
            if cand in used:
                continue
            test_set = current + [cand]
            if _check_all_conditions([(x, 1, y, 1) for (x, y) in test_set], 1):
                current.append(cand)
                used.add(cand)
        
        if len(current) < 3:
            continue
        
        # Greedy grow
        current = _greedy_grow(current, candidates, 1, time_limit, start_time)
        
        if len(current) > best_size:
            best_size = len(current)
            best = list(current)
        
        if best_size >= 8:
            break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Start with the known 5-point solution
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = list(base_points)
    best_D = 1

    # Try to extend with more points using integer search with incremental checks
    if time.time() < deadline - 30:
        extended = _search_integer_points(
            max_coord=400,
            max_points=8,
            time_limit=min(60, deadline - time.time() - 30)
        )
        if len(extended) > len(best):
            best = extended
            best_D = 1

    # Try parametrization-based search
    if time.time() < deadline - 10 and len(best) < 8:
        param_result = _search_with_parametrization(
            time_limit=min(40, deadline - time.time() - 10)
        )
        if len(param_result) > len(best):
            best = param_result
            best_D = 1

    # Verify the best solution
    pts = [(x, 1, y, 1) for (x, y) in best]
    if not _check_all_conditions(pts, best_D):
        # Fallback to base
        best = list(base_points)
        best_D = 1

    return {
        'D': best_D,
        'points': [[[x, 1], [y, 1]] for (x, y) in best]
    }
# EVOLVE-BLOCK-END
