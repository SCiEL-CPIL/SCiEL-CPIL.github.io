import time
import math
from fractions import Fraction
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 148.0  # 2s safety margin before 150s limit

    def is_rational_sqrt(num, den):
        """Check if num/den is a perfect square of a rational."""
        if num == 0:
            return True
        if den < 0:
            num = -num
            den = -den
        g = math.gcd(abs(num), abs(den))
        n = abs(num) // g
        d = abs(den) // g
        if n == 0:
            return True
        root_n = math.isqrt(n)
        root_d = math.isqrt(d)
        if root_n * root_n != n or root_d * root_d != d:
            return False
        return True

    def dist_sq_is_rational_square(p1, p2, D):
        """Check if squared distance between p1 and p2 is a rational square."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + D * dy*dy
        if d2 < 0:
            return False
        return is_rational_sqrt(d2.numerator, d2.denominator)

    def check_no_3_collinear(points):
        """Check no 3 points are collinear in (x, y) coordinates."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Cross product in (x, y) space
                    val = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    if val == 0:
                        return False
        return True

    def check_no_4_concyclic(points, D):
        """Check no 4 points are concyclic.
        Four points (x_i, y_i*sqrt(D)) are concyclic iff there exist A,B,C,R such that
        (x_i)^2 + D*(y_i)^2 + A*x_i + B*y_i + C = R^2 for all four points.
        This is equivalent to the 4x4 determinant being zero:
        |x_i^2 + D*y_i^2, x_i, y_i, 1| for i=1..4
        """
        n = len(points)
        for combo in combinations(range(n), 4):
            p1, p2, p3, p4 = (points[i] for i in combo)
            # Build 4x4 matrix
            # Row i: [x_i^2 + D*y_i^2, x_i, y_i, 1]
            r1 = [p1[0]**2 + D*p1[1]**2, p1[0], p1[1], Fraction(1)]
            r2 = [p2[0]**2 + D*p2[1]**2, p2[0], p2[1], Fraction(1)]
            r3 = [p3[0]**2 + D*p3[1]**2, p3[0], p3[1], Fraction(1)]
            r4 = [p4[0]**2 + D*p4[1]**2, p4[0], p4[1], Fraction(1)]
            # Compute determinant of 4x4 matrix
            def det4(m):
                a,b,c,d = m[0]
                e,f,g,h = m[1]
                i,j,k,l = m[2]
                m_,n_,o,p = m[3]
                return (
                    a*(f*(k*p-l*o) - g*(j*p-l*n_) + h*(j*o-k*n_)) -
                    b*(e*(k*p-l*o) - g*(i*p-l*m_) + h*(i*o-k*m_)) +
                    c*(e*(j*p-l*n_) - f*(i*p-l*m_) + h*(i*n_-j*m_)) -
                    d*(e*(j*o-k*n_) - f*(i*o-k*m_) + g*(i*n_-j*m_))
                )
            if det4([r1, r2, r3, r4]) == 0:
                return False
        return True

    def is_valid_set(points, D):
        """Full validity check."""
        n = len(points)
        # Check distinctness
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
        # Check all pairwise distances are rational
        for i in range(n):
            for j in range(i+1, n):
                if not dist_sq_is_rational_square(points[i], points[j], D):
                    return False
        # Check no 3 collinear
        if not check_no_3_collinear(points):
            return False
        # Check no 4 concyclic
        if not check_no_4_concyclic(points, D):
            return False
        return True

    def to_output_format(points, D):
        return {
            'D': D,
            'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]
        }

    # Known 5-point set for D=1 (verified)
    known_5pts = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]

    best_result = to_output_format(known_5pts, 1)
    best_size = 5

    # Verify the known set
    if is_valid_set(known_5pts, 1):
        best_size = 5
    else:
        # Fallback to a simpler valid set
        simple_4pts = [
            (Fraction(0), Fraction(0)),
            (Fraction(3), Fraction(0)),
            (Fraction(0), Fraction(4)),
            (Fraction(5), Fraction(0)),
        ]
        if is_valid_set(simple_4pts, 1):
            best_result = to_output_format(simple_4pts, 1)
            best_size = 4

    def generate_circle_points(center, D, radius, num_params=200):
        """Generate points on circle of given radius around center using rational parameterization.
        For D=1: circle (x-a)^2 + (y-b)^2 = r^2
        Param: x = a + r*(1-t^2)/(1+t^2), y = b + 2*r*t/(1+t^2) for rational t
        For general D: (x-a)^2 + D*(y-b)^2 = r^2
        Using: u = s*(D*m^2-1)/(D*m^2+1), v = -2*m*s/(D*m^2+1)
        So: x = a + s*(D*m^2-1)/(D*m^2+1), y = b - 2*m*s/(D*m^2+1)
        """
        candidates = []
        # Generate many parameter values for better coverage
        for m_num in range(-40, 41):
            for m_den in range(1, 12):
                if m_den == 0:
                    continue
                m = Fraction(m_num, m_den)
                denom = D * m * m + 1
                if denom == 0:
                    continue
                x = center[0] + radius * (D * m * m - 1) / denom
                y = center[1] - 2 * m * radius / denom
                candidates.append((x, y))
        
        return candidates

    def greedy_expand(points, D, max_size=12):
        """Greedily add points to maximize set size."""
        current = list(points)
        if not current:
            return current
        
        while len(current) < max_size:
            if time.time() > deadline:
                break
            
            # Generate candidates: points on circles around existing points
            candidates = []
            for anchor in current:
                # Try several radii (rational distances)
                for r_num in range(1, 30):
                    for r_den in range(1, 8):
                        r = Fraction(r_num, r_den)
                        circ_pts = generate_circle_points(anchor, D, r, 50)
                        for cp in circ_pts:
                            # Check rational distance to all current points
                            valid = True
                            for a in current:
                                if not dist_sq_is_rational_square(cp, a, D):
                                    valid = False
                                    break
                            if valid:
                                candidates.append(cp)
            
            # Remove duplicates and existing points
            seen = set(current)
            unique_cands = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_cands.append(c)
            
            # Sort by distance to centroid (prefer points that spread out)
            if unique_cands:
                cx = sum(p[0] for p in current) / len(current)
                cy = sum(p[1] for p in current) / len(current)
                unique_cands.sort(key=lambda p: abs((p[0]-cx).numerator) + abs((p[1]-cy).numerator))
            
            added = False
            for cand in unique_cands:
                if time.time() > deadline:
                    break
                new_pts = current + [cand]
                if is_valid_set(new_pts, D):
                    current = new_pts
                    added = True
                    break
            
            if not added:
                # Try random candidates as fallback
                found = False
                for _ in range(300):
                    if time.time() > deadline:
                        break
                    x = Fraction(random.randint(-40, 40), random.randint(1, 12))
                    y = Fraction(random.randint(-40, 40), random.randint(1, 12))
                    cand = (x, y)
                    if cand in set(current):
                        continue
                    valid = True
                    for a in current:
                        if not dist_sq_is_rational_square(cand, a, D):
                            valid = False
                            break
                    if valid:
                        new_pts = current + [cand]
                        if is_valid_set(new_pts, D):
                            current = new_pts
                            found = True
                            break
                if not found:
                    break
        
        return current

    # Strategy 1: Expand the known 5-point set for D=1
    if time.time() < deadline - 5:
        result = greedy_expand(known_5pts, 1, max_size=12)
        if len(result) > best_size:
            best_size = len(result)
            best_result = to_output_format(result, 1)

    # Strategy 2: Try different starting sets for D=1
    if time.time() < deadline - 5:
        # Try a 4-point seed and expand
        seeds_d1 = [
            [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4)), (Fraction(3), Fraction(4))],
            [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(12)), (Fraction(5), Fraction(12))],
            [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)), (Fraction(1), Fraction(1))],
            [(Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)), (Fraction(0), Fraction(3)), (Fraction(4), Fraction(3))],
            [(Fraction(0), Fraction(0)), (Fraction(8), Fraction(0)), (Fraction(0), Fraction(6)), (Fraction(8), Fraction(6))],
        ]
        for seed in seeds_d1:
            if time.time() > deadline - 3:
                break
            if not is_valid_set(seed, 1):
                continue
            result = greedy_expand(seed, 1, max_size=12)
            if len(result) > best_size:
                best_size = len(result)
                best_result = to_output_format(result, 1)

    # Strategy 3: Try D=2, 3, etc.
    if time.time() < deadline - 5:
        for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
            if time.time() > deadline - 3:
                break
            
            # For D>1, start with simple configurations
            # (0,0) and (a,0) have distance |a| (rational)
            # (0,0) and (a,b): distance^2 = a^2 + D*b^2, need this to be a rational square.
            
            # Simple seed: two points on x-axis
            seed = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))]
            if is_valid_set(seed, D):
                result = greedy_expand(seed, D, max_size=12)
                if len(result) > best_size:
                    best_size = len(result)
                    best_result = to_output_format(result, D)
            
            # Try 3-point seeds for D>1 where we can find valid triangles
            # For D=2: (0,0), (2,0), (0,1) -> dist(0,0)-(0,1) = sqrt(2) not rational
            # But (0,0), (2,0), (1,1) -> dist^2 = 1+2=3 not square
            # (0,0), (1,0), (0,1) -> dist^2=1, 1, 2 -> 2 not square
            # For D=3: (0,0), (3,0), (0,1) -> dist^2=3 not square
            # (0,0), (1,0), (0,1) -> dist^2=1, 1, 3 -> 3 not square
            # For D=5: (0,0), (3,0), (0,1) -> dist^2=5 not square
            # (0,0), (1,0), (0,2) -> dist^2=1, 20, 21 -> 20,21 not squares
            # Let's just try the 2-point seed and let greedy expand.

    # Strategy 4: More aggressive expansion on best result
    if time.time() < deadline - 2 and best_size >= 4:
        D_best = best_result['D']
        pts_best = [(Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) 
                    for p in best_result['points']]
        improved = greedy_expand(pts_best, D_best, max_size=12)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = to_output_format(improved, D_best)

    return best_result
