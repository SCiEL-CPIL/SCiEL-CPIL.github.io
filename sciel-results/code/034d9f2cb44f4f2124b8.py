import math
from fractions import Fraction
import time
import itertools

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
    """Compute sqrt of a Fraction that is a perfect square."""
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

def _try_extend_systematic(points, D, deadline, max_radius_num=200, max_radius_den=50):
    """Try to add one more point using systematic radius search."""
    start = time.time()
    P = list(points)
    n = len(P)
    
    # Precompute all pairwise squared distances
    pair_dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_dists[(i, j)] = _dist_sq(P[i], P[j], D)
    
    # Generate radius candidates: r^2 values where r is rational
    radius_candidates = []
    for num in range(1, max_radius_num + 1):
        for den in range(1, max_radius_den + 1):
            if math.gcd(num, den) == 1:  # Only reduced fractions
                r = Fraction(num, den)
                r2 = r * r
                radius_candidates.append(r2)
    
    # Sort by value for early termination
    radius_candidates.sort()
    
    # Try all pairs of reference points
    ref_pairs = list(itertools.combinations(range(n), 2))
    
    for (i, j) in ref_pairs:
        if time.time() > deadline - 10:
            break
        P_i = P[i]
        P_j = P[j]
        d2_ij = pair_dists[(i, j)]
        
        if not _is_square_frac(d2_ij):
            continue
        d_ij = _sqrt_frac(d2_ij)
        
        # For each pair of radii
        for r1_sq in radius_candidates:
            if time.time() > deadline - 10:
                break
            for r2_sq in radius_candidates:
                if time.time() > deadline - 10:
                    break
                
                # Triangle inequality check
                r1 = _sqrt_frac(r1_sq)
                r2 = _sqrt_frac(r2_sq)
                if r1 + r2 < d_ij or r1 + d_ij < r2 or r2 + d_ij < r1:
                    continue
                if r1 + r2 == d_ij or r1 + d_ij == r2 or r2 + d_ij == r1:
                    continue
                
                # Intersection of two circles centered at P_i and P_j
                # x = x_i + (r1^2 - r2^2 + d^2) / (2*d)
                x_offset = (r1_sq - r2_sq + d2_ij) / (2 * d_ij)
                x = P_i[0] + x_offset
                
                # (y - y_i)^2 = r1^2 - x_offset^2
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
                    new_set = P + [Q]
                    if not _no_three_collinear(new_set):
                        continue
                    
                    # Check no four concyclic with new point
                    if not _no_four_concyclic(new_set):
                        continue
                    
                    return new_set
    
    return None

def _search_integer_points_6(deadline):
    """Search for 6 integer points with integer pairwise distances."""
    # Start with known 5-point set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    P = [(Fraction(x), Fraction(y)) for x, y in base5]
    D = 1
    
    if not _validate(P, D):
        return None
    
    # Try to extend to 6 points
    extended = _try_extend_systematic(P, D, deadline, max_radius_num=300, max_radius_den=60)
    if extended:
        return extended
    
    # Try different 5-point bases
    # Generate random 5-point sets and try to extend
    import random
    random.seed(12345)
    
    for attempt in range(50):
        if time.time() > deadline - 15:
            break
        
        # Generate a random 5-point set with integer coordinates
        pts = []
        for _ in range(5):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            pts.append((x, y))
        
        # Check if all pairwise distances are integers
        valid = True
        for i in range(5):
            for j in range(i + 1, 5):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d2 = dx * dx + dy * dy
                if not _is_square_frac(Fraction(d2)):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        P_frac = [(Fraction(x), Fraction(y)) for x, y in pts]
        if not _validate(P_frac, 1):
            continue
        
        extended = _try_extend_systematic(P_frac, 1, deadline, max_radius_num=200, max_radius_den=40)
        if extended:
            return extended
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Known 5-point set
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    points = [(Fraction(x), Fraction(y)) for x, y in known_5]
    
    best_points = points
    best_D = D
    
    if _validate(points, D):
        best_points = points
        best_D = D
    
    # Try to extend to 6 points
    if time.time() < deadline - 20:
        extended = _try_extend_systematic(best_points, best_D, deadline, max_radius_num=400, max_radius_den=80)
        if extended:
            best_points = extended
            best_D = best_D
    
    # If still 5 points, try other strategies
    if len(best_points) < 6 and time.time() < deadline - 30:
        # Try searching for 6-point sets from scratch
        result = _search_integer_points_6(deadline)
        if result and len(result) > len(best_points):
            best_points = result
            best_D = 1
    
    # Try other D values
    if len(best_points) < 6 and time.time() < deadline - 15:
        for D_test in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 21, 22, 26, 30, 33, 34, 35, 38, 39, 42, 46, 47, 51, 55, 57, 58, 59, 62, 65, 66, 69, 70, 74, 77, 78, 79, 82, 85, 86, 87, 90, 91, 93, 94, 95]:
            if time.time() > deadline - 10:
                break
            
            # Quick check: find a 3-point set for this D
            base = [(Fraction(0), Fraction(0))]
            found_3 = None
            
            for x1 in range(1, 30):
                if time.time() > deadline - 10:
                    break
                for y2 in range(1, 30):
                    if time.time() > deadline - 10:
                        break
                    p1 = (Fraction(x1), Fraction(0))
                    p2 = (Fraction(0), Fraction(y2))
                    
                    d2_02 = Fraction(D_test) * y2 * y2
                    if not _is_square_frac(d2_02):
                        continue
                    d12 = x1 * x1 + D_test * y2 * y2
                    if not _is_square_frac(Fraction(d12)):
                        continue
                    
                    candidate_3 = [base[0], p1, p2]
                    if _validate(candidate_3, D_test):
                        found_3 = candidate_3
                        break
                if found_3:
                    break
            
            if found_3 is None:
                continue
            
            # Try to extend
            current = found_3
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                extended = _try_extend_systematic(current, D_test, deadline, max_radius_num=100, max_radius_den=20)
                if extended:
                    current = extended
                else:
                    break
            
            if len(current) > len(best_points):
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
