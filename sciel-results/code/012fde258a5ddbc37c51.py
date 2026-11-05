import math
from fractions import Fraction
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _is_square_frac(f):
    """Check if Fraction f is a perfect square of a rational."""
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return rn * rn == n and rd * rd == d

def _sqrt_frac(f):
    """Compute sqrt of a Fraction that is known to be a perfect square."""
    n, d = f.numerator, f.denominator
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return Fraction(rn, rd)

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x,y) in the (x, y*sqrt(D)) metric."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + Fraction(D) * dy * dy

def _is_rational_dist(p1, p2, D):
    d2 = _dist_sq(p1, p2, D)
    return _is_square_frac(d2)

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    M = [
                        [p1[0], p1[1], p1[0]**2 + p1[1]**2, Fraction(1)],
                        [p2[0], p2[1], p2[0]**2 + p2[1]**2, Fraction(1)],
                        [p3[0], p3[1], p3[0]**2 + p3[1]**2, Fraction(1)],
                        [p4[0], p4[1], p4[0]**2 + p4[1]**2, Fraction(1)],
                    ]
                    def det4(M):
                        return (M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]))
                             - M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]))
                             + M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0]))
                             - M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])))
                    if det4(M) == 0:
                        return False
    return True

def _no_four_concyclic_with_new(P, Q):
    """Check that no four points among P + [Q] are concyclic, assuming P has no 4 concyclic."""
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3, p4 = P[i], P[j], P[k], Q
                M = [
                    [p1[0], p1[1], p1[0]**2 + p1[1]**2, Fraction(1)],
                    [p2[0], p2[1], p2[0]**2 + p2[1]**2, Fraction(1)],
                    [p3[0], p3[1], p3[0]**2 + p3[1]**2, Fraction(1)],
                    [p4[0], p4[1], p4[0]**2 + p4[1]**2, Fraction(1)],
                ]
                def det4(M):
                    return (M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]))
                         - M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]))
                         + M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0]))
                         - M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])))
                if det4(M) == 0:
                    return False
    return True

def _no_three_collinear_with_new(P, Q):
    """Check that no three points among P + [Q] are collinear, assuming P has no 3 collinear."""
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2, p3 = P[i], P[j], Q
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if cross == 0:
                return False
    return True

def _validate(points, D):
    if len(points) < 3:
        return False
    if len(set(points)) != len(points):
        return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if not _is_rational_dist(points[i], points[j], D):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _try_extend(points, D, deadline, max_iterations=50000):
    """Try to add one more point to the given set."""
    start = time.time()
    P = list(points)
    n = len(P)
    
    # Precompute all pairwise distances (squared) for reference
    pair_dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_dists[(i, j)] = _dist_sq(P[i], P[j], D)
    
    # Generate diverse radius candidates
    radius_candidates = []
    for num in range(1, 60):
        for den in range(1, 20):
            if math.gcd(num, den) == 1:
                r = Fraction(num, den)
                r2 = r * r
                radius_candidates.append((r2, r))
    
    # Sort by radius for diversity
    radius_candidates.sort(key=lambda x: float(x[0]))
    
    # Take a subset for speed
    if len(radius_candidates) > 400:
        # Sample evenly
        step = len(radius_candidates) // 400
        radius_candidates = radius_candidates[::step]
    
    ref_pairs = list(itertools.combinations(range(n), 2))
    
    iterations = 0
    for (i, j) in ref_pairs:
        if time.time() > deadline - 5:
            break
        iterations += 1
        P_i = P[i]
        P_j = P[j]
        d2_ij = pair_dists[(i, j)]
        
        if not _is_square_frac(d2_ij):
            continue
        d_ij = _sqrt_frac(d2_ij)
        
        for (r1_sq, r1) in radius_candidates:
            if time.time() > deadline - 5:
                break
            iterations += 1
            for (r2_sq, r2) in radius_candidates:
                if time.time() > deadline - 5:
                    break
                iterations += 1
                
                # Check triangle inequality roughly
                if r1 + r2 < d_ij or r1 + d_ij < r2 or r2 + d_ij < r1:
                    continue
                if r1 + r2 == d_ij or r1 + d_ij == r2 or r2 + d_ij == r1:
                    continue
                
                # Intersection of two circles
                x_offset = (r1_sq - r2_sq + d2_ij) / (2 * d_ij)
                x = P_i[0] + x_offset
                
                y_sq = r1_sq - x_offset * x_offset
                if y_sq < 0:
                    continue
                if not _is_square_frac(y_sq):
                    continue
                y_offset = _sqrt_frac(y_sq)
                
                candidates = [(x, P_i[1] + y_offset), (x, P_i[1] - y_offset)]
                
                for qx, qy in candidates:
                    Q = (qx, qy)
                    if Q in P:
                        continue
                    
                    # Check rational distance to all existing points
                    valid = True
                    for pk in P:
                        if not _is_rational_dist(Q, pk, D):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check no three collinear with new point
                    if not _no_three_collinear_with_new(P, Q):
                        continue
                    
                    # Check no four concyclic with new point
                    if not _no_four_concyclic_with_new(P, Q):
                        continue
                    
                    return P + [Q]
    
    return None

def _find_3point_base(D, deadline):
    """Find a 3-point base set for given D."""
    base = [(Fraction(0), Fraction(0))]
    
    for x1_num in range(1, 50):
        if time.time() > deadline - 10:
            break
        for y2_num in range(1, 50):
            if time.time() > deadline - 10:
                break
            p1 = (Fraction(x1_num), Fraction(0))
            p2 = (Fraction(0), Fraction(y2_num))
            
            d2 = Fraction(D) * y2_num * y2_num
            if not _is_square_frac(d2):
                continue
            d12 = x1_num * x1_num + D * y2_num * y2_num
            if not _is_square_frac(Fraction(d12)):
                continue
            
            candidate_3 = [base[0], p1, p2]
            if _validate(candidate_3, D):
                return candidate_3
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_D = None
    
    # Known 5-point set with D=1
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    points = [(Fraction(x), Fraction(y)) for x, y in known_5]
    
    if _validate(points, D):
        best_points = points
        best_D = D
    
    # Try to extend to 6 points with the known 5-point set
    if best_points is not None and time.time() < deadline - 30:
        extended = _try_extend(best_points, best_D, deadline)
        if extended:
            best_points = extended
            best_D = best_D
    
    # Try other D values
    for D_test in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97]:
        if time.time() > deadline - 15:
            break
        
        found_3 = _find_3point_base(D_test, deadline)
        if found_3 is None:
            continue
        
        current = found_3
        for _ in range(4):  # Try to add up to 4 more points
            if time.time() > deadline - 15:
                break
            extended = _try_extend(current, D_test, deadline)
            if extended:
                current = extended
            else:
                break
        
        if len(current) > (len(best_points) if best_points else 0):
            best_points = current
            best_D = D_test
    
    # Format output
    if best_points is None:
        best_points = [(Fraction(x), Fraction(y)) for x, y in known_5]
        best_D = 1
    
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
