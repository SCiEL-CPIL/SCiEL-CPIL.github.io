"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_dist_rational(p1, p2, D):
    """Check if squared distance is a perfect square of a rational number."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    # dist^2 = dx^2 + D * dy^2
    # x, y are Fractions. dx, dy are Fractions.
    # dx^2 + D*dy^2 must be a square of a rational.
    d2 = dx*dx + D*dy*dy
    if d2 < 0:
        return False
    # d2 is a Fraction. Let d2 = num/den.
    # We need sqrt(num/den) to be rational.
    # This is equivalent to num*den being a perfect square integer.
    num = d2.numerator
    den = d2.denominator
    prod = num * den
    if prod == 0:
        return True
    return _is_square(prod)

def _check_collinear(p1, p2, p3):
    """Check if three points are collinear. Points are (x, y) where geometric y is y*sqrt(D).
    Collinearity in geometric space:
    (x1-x2)*(y3-y2)*sqrt(D) - (y1-y2)*sqrt(D)*(x3-x2) = 0
    => (x1-x2)*(y3-y2) - (y1-y2)*(x3-x2) = 0
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    cross = (x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2)
    return cross == 0

def _check_concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic.
    Circle equation in geometric coords (X, Y):
    X^2 + Y^2 + AX + BY + C = 0
    Substituting X=x, Y=y*sqrt(D):
    x^2 + D*y^2 + Ax + B*y*sqrt(D) + C = 0
    
    For rational x, y and integer D, this equation holds if and only if the coefficients of 1 and sqrt(D) are zero?
    Actually, we are looking for a circle in the geometric plane.
    The condition for 4 points to be concyclic is that the determinant of the matrix:
    | x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1 | is 0.
    
    Let's use the property that 4 points (x_i, y_i) are concyclic in the metric space if they satisfy a generalized circle equation.
    However, a simpler check for integer lattice points (D=1) is standard determinant.
    For general D, we can transform coordinates? No.
    
    Let's use the determinant expansion.
    Let a_i = x_i^2 + D*y_i^2.
    Matrix M:
    [ a1, x1, y1*sqrt(D), 1 ]
    [ a2, x2, y2*sqrt(D), 1 ]
    [ a3, x3, y3*sqrt(D), 1 ]
    [ a4, x4, y4*sqrt(D), 1 ]
    
    det(M) = 0.
    Expand along last column:
    det = 1*det(M1) - 1*det(M2) + 1*det(M3) - 1*det(M4)
    where M1 is submatrix removing row 1 col 4, etc.
    
    Each 3x3 submatrix has form:
    [ a_i, x_i, y_i*sqrt(D) ]
    [ a_j, x_j, y_j*sqrt(D) ]
    [ a_k, x_k, y_k*sqrt(D) ]
    
    det_sub = a_i(x_j y_k sqrt(D) - x_k y_j sqrt(D)) - x_i(a_j y_k sqrt(D) - a_k y_j sqrt(D)) + y_i sqrt(D)(a_j x_k - a_k x_j)
    = sqrt(D) * [ a_i(x_j y_k - x_k y_j) - x_i(a_j y_k - a_k y_j) + y_i(a_j x_k - a_k x_j) ]
    
    Let R(i,j,k) = a_i(x_j y_k - x_k y_j) - x_i(a_j y_k - a_k y_j) + y_i(a_j x_k - a_k x_j)
    Then det(M) = sqrt(D) * ( R(2,3,4) - R(1,3,4) + R(1,2,4) - R(1,2,3) )
    
    So det(M) = 0 iff R(2,3,4) - R(1,3,4) + R(1,2,4) - R(1,2,3) = 0.
    """
    pts = [p1, p2, p3, p4]
    a = [p[0]**2 + D*p[1]**2 for p in pts]
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    
    def R(i, j, k):
        return a[i]*(x[j]*y[k] - x[k]*y[j]) - x[i]*(a[j]*y[k] - a[k]*y[j]) + y[i]*(a[j]*x[k] - a[k]*x[j])
        
    # Indices 0,1,2,3
    val = R(1, 2, 3) - R(0, 2, 3) + R(0, 1, 3) - R(0, 1, 2)
    return val == 0

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Known 5-point set for D=1
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Let's verify distances are integers.
    # d((0,0), (-56,33)) = sqrt(56^2+33^2) = sqrt(3136+1089) = sqrt(4225) = 65. OK.
    # d((0,0), (-16,63)) = sqrt(16^2+63^2) = sqrt(256+3969) = sqrt(4225) = 65. OK.
    # d((0,0), (0,33)) = 33. OK.
    # d((0,0), (16,63)) = 65. OK.
    # d((-56,33), (-16,63)) = sqrt(40^2+30^2) = 50. OK.
    # d((-56,33), (0,33)) = 56. OK.
    # d((-56,33), (16,63)) = sqrt(72^2+30^2) = sqrt(5184+900) = sqrt(6084) = 78. OK.
    # d((-16,63), (0,33)) = sqrt(16^2+30^2) = sqrt(256+900) = sqrt(1156) = 34. OK.
    # d((-16,63), (16,63)) = 32. OK.
    # d((0,33), (16,63)) = sqrt(16^2+30^2) = 34. OK.
    
    D = 1
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    
    best_points = base_points
    best_score = len(base_points)
    
    # Generate candidates
    # We search in a box around the base points.
    # Base points are in [-56, 16] x [0, 63].
    # Let's expand to [-100, 100] x [-100, 100]
    
    # To make search faster, we only consider integer points.
    # We can pre-filter candidates that have integer distance to at least one base point.
    
    candidates = []
    # We will use a set to avoid duplicates
    seen = set(base_points)
    
    # Optimization: Pre-compute squared distances from base points?
    # No, just iterate.
    
    # We want to find points that can be added.
    # Strategy: Greedy search from different starting points.
    
    # 1. Start with the full base set and try to add.
    # 2. Start with subsets of base set and try to add.
    
    # Candidate generation
    # We need to be careful with time.
    # Let's generate a list of potential candidates in a larger box.
    min_x, max_x = -150, 150
    min_y, max_y = -150, 150
    
    # We will not generate all 300x300=90000 points if it's too slow to check.
    # But checking distance to 5 points is fast.
    # Checking validity of adding to a set of 5 is O(5^2 + 5^3) = 25 + 125 = 150 ops.
    # 90000 * 150 = 13.5M ops. Should be fast in Python? Maybe 1-2 seconds.
    
    # Let's refine candidate list: only points with integer distance to at least one base point.
    valid_candidates = []
    for x in range(min_x, max_x + 1):
        if time.time() > deadline:
            break
        for y in range(min_y, max_y + 1):
            p = (Fraction(x), Fraction(y))
            if p in seen:
                continue
            # Check if dist to any base point is integer
            has_int_dist = False
            for bp in base_points:
                if _check_dist_rational(p, bp, D):
                    has_int_dist = True
                    break
            if has_int_dist:
                valid_candidates.append(p)
        
        if time.time() > deadline:
            break
            
    # Shuffle candidates for variety
    random.seed(42)
    random.shuffle(valid_candidates)
    
    # Greedy Add
    def try_greedy(start_set, candidates_list, max_size=12):
        current = list(start_set)
        current_set = set(current)
        
        # Pre-check start_set validity
        # Distances
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                if not _check_dist_rational(current[i], current[j], D):
                    return current
        # Collinear
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                for k in range(j+1, len(current)):
                    if _check_collinear(current[i], current[j], current[k]):
                        return current
        # Concyclic
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                for k in range(j+1, len(current)):
                    for l in range(k+1, len(current)):
                        if _check_concyclic(current[i], current[j], current[k], current[l], D):
                            return current
                            
        while len(current) < max_size:
            added = False
            # Try to add one candidate
            for cand in candidates_list:
                if cand in current_set:
                    continue
                if time.time() > deadline:
                    return current
                
                # Check distance to all existing
                ok = True
                for p in current:
                    if not _check_dist_rational(cand, p, D):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check collinearity with all pairs
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if _check_collinear(cand, current[i], current[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                # Check concyclicity with all triples
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        for k in range(j+1, len(current)):
                            if _check_concyclic(cand, current[i], current[j], current[k], D):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                # All checks passed
                current.append(cand)
                current_set.add(cand)
                added = True
                break # Restart loop to re-evaluate order? Or continue?
                # Restarting allows picking the "best" next point.
            
            if not added:
                break
                
        return current

    # Strategy 1: Start with full base set
    result1 = try_greedy(base_points, valid_candidates)
    if len(result1) > best_score:
        best_score = len(result1)
        best_points = result1
        if best_score >= 8:
            break # Found new record? Actually we want max, but 8 is huge.
            # Let's keep searching if time permits.

    # Strategy 2: Try subsets of base points to see if we can find a different, larger set
    # This is expensive. We will only do this if time remains.
    
    if time.time() < deadline - 20:
        # Try removing one point from base and extending
        for i in range(len(base_points)):
            subset = base_points[:i] + base_points[i+1:]
            # Ensure subset is valid (it is, as it's a subset of valid set)
            res = try_greedy(subset, valid_candidates)
            if len(res) > best_score:
                best_score = len(res)
                best_points = res
    
    # If we didn't find > 5, return base
    # Convert Fractions to [num, den]
    
    final_points = []
    for p in best_points:
        x, y = p
        final_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
        
    return {'D': D, 'points': final_points}

# EVOLVE-BLOCK-END
