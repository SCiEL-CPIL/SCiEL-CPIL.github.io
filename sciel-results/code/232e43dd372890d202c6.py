"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt, gcd
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Try known 5-point construction first
    best_result = _try_known_5_points()
    
    if time.time() > deadline - 3:
        return best_result
    
    # Try to find 6+ points with D=1 using smarter search
    result_6 = _search_6_points_smart()
    if result_6 is not None:
        return result_6
    
    if time.time() > deadline - 2:
        return best_result
    
    # Try D > 1 constructions with focused search
    result_d = _search_with_D_focused()
    if result_d is not None and _count_points(result_d) > _count_points(best_result):
        return result_d
    
    return best_result

def _count_points(result):
    return len(result['points'])

def _try_known_5_points():
    # Five integer points with integer pairwise distances, no three collinear and no four concyclic
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_pairwise_distances(points, D):
    """Check all pairwise distances are rational (squared distances are perfect squares when scaled)"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + D*dy*dy
            if not _is_square(dist_sq):
                return False
    return True

def _no_three_collinear(points):
    """Check no three points are collinear"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _four_points_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _four_points_concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic using determinant method"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = -(x2*x2 - x1*x1) - D*(y2*y2 - y1*y1)
    
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = -(x3*x3 - x1*x1) - D*(y3*y3 - y1*y1)
    
    det = a1*b2 - a2*b1
    if det == 0:
        return False
    
    rhs = x1*x1 + D*y1*y1 - x4*x4 - D*y4*y4
    
    lhs = (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1)
    rhs_scaled = det * rhs
    
    return lhs == rhs_scaled

def _validate(points, D):
    """Full validation of a point set"""
    if len(points) < 3:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    if not _check_pairwise_distances(points, D):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _search_6_points_smart():
    """Smart search for 6 integer points with integer pairwise distances"""
    # Generate candidate points with integer distances from origin
    candidates = set()
    max_coord = 50
    
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if x == 0 and y == 0:
                continue
            dist_sq = x*x + y*y
            if _is_square(dist_sq):
                candidates.add((x, y))
    
    candidates = list(candidates)
    random.seed(123)
    
    # Try random 6-point subsets
    for _ in range(2000):
        if len(candidates) < 6:
            break
        subset = random.sample(candidates, 6)
        if _validate(subset, 1):
            return {'D': 1, 'points': [[[px, 1], [py, 1]] for px, py in subset]}
    
    # Try greedy approach: build up point set one by one
    for _ in range(100):
        if len(candidates) < 6:
            break
        points = [random.choice(candidates)]
        remaining = set(candidates) - {points[0]}
        
        for _ in range(5):
            if not remaining:
                break
            found = False
            for p in random.sample(list(remaining), min(50, len(remaining))):
                if _validate(points + [p], 1):
                    points.append(p)
                    remaining.remove(p)
                    found = True
                    break
            if not found:
                break
        
        if len(points) >= 6:
            return {'D': 1, 'points': [[[px, 1], [py, 1]] for px, py in points]}
    
    return None

def _search_with_D_focused():
    """Focused search for point sets with D > 1"""
    # Try small squarefree D values with rational coordinates
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31]:
        # Try simple rational configurations
        # Points with small rational coordinates
        candidates = []
        for x_num in range(-5, 6):
            for x_den in [1, 2, 3]:
                for y_num in range(-5, 6):
                    for y_den in [1, 2, 3]:
                        candidates.append((Fraction(x_num, x_den), Fraction(y_num, y_den)))
        
        candidates = list(set(candidates))
        random.seed(456)
        
        for _ in range(500):
            if len(candidates) < 6:
                break
            subset = random.sample(candidates, 6)
            # Convert to integer tuples for validation
            int_points = [(int(x), int(y)) for x, y in subset if x.denominator == 1 and y.denominator == 1]
            if len(int_points) >= 6 and _validate(int_points, D):
                return {'D': D, 'points': [[[int(x), 1], [int(y), 1]] for x, y in int_points]}
        
        # Try with all rational points (not just integers)
        for _ in range(200):
            if len(candidates) < 6:
                break
            subset = random.sample(candidates, 6)
            # Check distances with rational arithmetic
            if _validate_rational(subset, D):
                return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in subset]}
    
    return None

def _validate_rational(points, D):
    """Validate points with rational coordinates"""
    if len(points) < 3:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    
    # Check pairwise distances
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + Fraction(D) * dy*dy
            # Check if dist_sq is a perfect square of a rational
            if dist_sq < 0:
                return False
            # dist_sq should be (a/b)^2 for some integers a, b
            # i.e., dist_sq * b^2 = a^2 for some b
            # Equivalently, dist_sq should be a square in Q
            num = dist_sq.numerator
            den = dist_sq.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                if cross == 0:
                    return False
    
    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _four_points_concyclic_rational(points[i], points[j], points[k], points[l], D):
                        return False
    
    return True

def _four_points_concyclic_rational(p1, p2, p3, p4, D):
    """Check if four points with rational coordinates are concyclic"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = -(x2*x2 - x1*x1) - Fraction(D)*(y2*y2 - y1*y1)
    
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = -(x3*x3 - x1*x1) - Fraction(D)*(y3*y3 - y1*y1)
    
    det = a1*b2 - a2*b1
    if det == 0:
        return False
    
    rhs = x1*x1 + Fraction(D)*y1*y1 - x4*x4 - Fraction(D)*y4*y4
    
    lhs = (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1)
    rhs_scaled = det * rhs
    
    return lhs == rhs_scaled
# EVOLVE-BLOCK-END
