"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Seed points (D=1)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = seed_points
    
    def is_square(n):
        if n < 0:
            return False
        r = int(round(n**0.5))
        return r*r == n or (r+1)*(r+1) == n

    def check_rational_dist(p1, p2, D):
        # p1, p2 are (x, y) integers
        # dist^2 = (x1-x2)^2 + D(y1-y2)^2
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + D*dy*dy
        return is_square(d2)

    def check_no_three_collinear(points, D):
        # Collinearity in (x, y*sqrt(D)) plane
        # Area = 0.5 * | x1(y2*sqrt(D) - y3*sqrt(D)) + ... |
        # = 0.5 * sqrt(D) * | x1(y2-y3) + x2(y3-y1) + x3(y1-y2) |
        # So check if x1(y2-y3) + x2(y3-y1) + x3(y1-y2) == 0
        for p1, p2, p3 in combinations(points, 3):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            area2 = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
            if area2 == 0:
                return False
        return True

    def check_no_four_concyclic(points, D):
        # 4 points (x1, y1), ..., (x4, y4) in (x, Y) plane where Y=y*sqrt(D)
        # Circle: X^2 + Y^2 + AX + BY + C = 0
        # x^2 + D y^2 + A x + B sqrt(D) y + C = 0
        # Let B' = B sqrt(D). Then x^2 + D y^2 + A x + B' y + C = 0
        # We need to check if there exist A, B', C (rationals) satisfying this for 4 points.
        # Since x, y, D are integers, if a solution exists, it is rational.
        # We can solve for A, B', C using 3 points and check the 4th.
        # However, B' might not be rational if D is not a square?
        # Wait, the points are in the plane. The circle coefficients A, B, C must be real.
        # For the points to be concyclic, the determinant of the circle matrix must be 0.
        # Matrix:
        # | x1^2 + D y1^2  x1  y1  1 |
        # | x2^2 + D y2^2  x2  y2  1 |
        # | x3^2 + D y3^2  x3  y3  1 |
        # | x4^2 + D y4^2  x4  y4  1 |
        # Note: The term is Y^2 = D y^2. The linear term is B Y = B sqrt(D) y.
        # So the column for Y is y, but the coefficient is B sqrt(D).
        # If we treat the system as linear in A, (B sqrt(D)), C, the variables are A, B_s, C.
        # The matrix entries are integers.
        # So we just check if the determinant is 0.
        
        for p1, p2, p3, p4 in combinations(points, 4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            # Determinant of 4x4 matrix
            # | x1^2+D y1^2  x1  y1  1 |
            # ...
            # We can compute this exactly.
            
            def det4(m):
                # m is 4x4 list of lists
                return (
                    m[0][0]*(m[1][1]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2] - m[2][2]*m[3][1])) -
                    m[0][1]*(m[1][0]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2] - m[2][2]*m[3][0])) +
                    m[0][2]*(m[1][0]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])) -
                    m[0][3]*(m[1][0]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1] - m[2][1]*m[3][0]))
                )
            
            m = [
                [x1*x1 + D*y1*y1, x1, y1, 1],
                [x2*x2 + D*y2*y2, x2, y2, 1],
                [x3*x3 + D*y3*y3, x3, y3, 1],
                [x4*x4 + D*y4*y4, x4, y4, 1]
            ]
            if det4(m) == 0:
                return False
        return True

    def is_valid_set(points, D):
        if len(points) < 3:
            return False
        # Check distinct
        if len(set(points)) != len(points):
            return False
        # Check distances
        for p1, p2 in combinations(points, 2):
            if not check_rational_dist(p1, p2, D):
                return False
        # Check collinearity
        if not check_no_three_collinear(points, D):
            return False
        # Check concyclicity
        if not check_no_four_concyclic(points, D):
            return False
        return True

    # Try to find 6 points with D=1
    # Search space: small integers
    # We will use a random restart local search or simple random sampling
    
    max_coord = 50
    found_6 = None
    
    # Pre-generate candidates with rational distances from origin?
    # No, just random points and check pairwise.
    
    attempts = 0
    while time.time() < deadline and attempts < 2000:
        attempts += 1
        # Generate 6 random points
        pts = []
        for _ in range(6):
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            if (x, y) not in pts:
                pts.append((x, y))
        
        if len(pts) < 6:
            continue
            
        if is_valid_set(pts, 1):
            found_6 = pts
            break
    
    if found_6:
        best_points = found_6
    else:
        # Try to improve the seed? Or just return seed.
        # The seed is valid.
        pass

    # Format output
    D = 1
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
