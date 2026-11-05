"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Helper functions
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
    
    def make_frac(n, d):
        """Create a simplified fraction from numerator and denominator."""
        if d == 0:
            return None
        if d < 0:
            n, d = -n, -d
        g = gcd(n, d)
        if g > 0:
            n //= g
            d //= g
        return (n, d)
    
    def frac_to_tuple(frac):
        """Convert Fraction to (num, den) tuple."""
        return (frac.numerator, frac.denominator)
    
    def point_dist_sq(p1, p2, D):
        """Squared distance between two points (x1,y1) and (x2,y2) where actual y is y*sqrt(D).
        p1, p2 are (x_num, x_den, y_num, y_den) tuples.
        Returns squared distance as Fraction."""
        x1 = Fraction(p1[0], p1[1])
        y1 = Fraction(p1[2], p1[3])
        x2 = Fraction(p2[0], p2[1])
        y2 = Fraction(p2[2], p2[3])
        
        dx = x1 - x2
        dy = y1 - y2
        d2 = dx * dx + D * dy * dy
        return d2
    
    def is_rational_distance(p1, p2, D):
        """Check if distance between p1 and p2 is rational."""
        d2 = point_dist_sq(p1, p2, D)
        # d2 must be a perfect square of a rational number
        # d2 = n/d in lowest terms, need n and d to be perfect squares
        if d2 < 0:
            return False
        if d2 == 0:
            return True
        n = d2.numerator
        d = d2.denominator
        return is_perfect_square(n) and is_perfect_square(d)
    
    def are_collinear(p1, p2, p3, D):
        """Check if three points are collinear using exact arithmetic."""
        x1 = Fraction(p1[0], p1[1])
        y1 = Fraction(p1[2], p1[3])
        x2 = Fraction(p2[0], p2[1])
        y2 = Fraction(p2[2], p2[3])
        x3 = Fraction(p3[0], p3[1])
        y3 = Fraction(p3[2], p3[3])
        
        # Collinearity: (x2-x1)(y3-y1) = (x3-x1)(y2-y1)
        left = (x2 - x1) * (y3 - y1)
        right = (x3 - x1) * (y2 - y1)
        return left == right
    
    def are_concyclic(p1, p2, p3, p4, D):
        """Check if four points are concyclic using exact arithmetic."""
        # Use the determinant test for concyclicity
        x1 = Fraction(p1[0], p1[1])
        y1 = Fraction(p1[2], p1[3])
        x2 = Fraction(p2[0], p2[1])
        y2 = Fraction(p2[2], p2[3])
        x3 = Fraction(p3[0], p3[1])
        y3 = Fraction(p3[2], p3[3])
        x4 = Fraction(p4[0], p4[1])
        y4 = Fraction(p4[2], p4[3])
        
        # Convert to actual coordinates: (x, y*sqrt(D))
        # For concyclicity test, we work with the actual geometric positions
        # A circle equation: x^2 + y^2 + Ax + By + C = 0
        # where y is the actual y-coordinate = y_coord * sqrt(D)
        
        # Let's use a different approach: four points (xi, yi*sqrt(D)) are concyclic
        # iff there exists a circle through all four.
        
        # Use the determinant method with actual coordinates
        # Point i has coordinates (xi, yi*sqrt(D))
        
        # Circle through 3 points, check if 4th lies on it
        # General circle: x^2 + y^2 + Dx + Ey + F = 0
        # For point (xi, yi*sqrt(D)): xi^2 + yi^2*D + D*xi + E*yi*sqrt(D) + F = 0
        
        # This gets complicated with sqrt(D). Let me use a different approach.
        
        # Four points are concyclic iff the cross ratio is real, or equivalently,
        # the determinant of the matrix below is zero:
        # | x1^2 + y1^2*D  x1  y1*sqrt(D)  1 |
        # | x2^2 + y2^2*D  x2  y2*sqrt(D)  1 |
        # | x3^2 + y3^2*D  x3  y3*sqrt(D)  1 |
        # | x4^2 + y4^2*D  x4  y4*sqrt(D)  1 |
        
        # But this has sqrt(D) terms. Let me multiply through or use a different test.
        
        # Alternative: use the fact that four points are concyclic iff
        # the angle subtended by a chord is the same.
        
        # Actually, let me just check if all four points lie on some circle.
        # A circle in the (x, y*sqrt(D)) plane: (x-h)^2 + (y*sqrt(D)-k)^2 = r^2
        # x^2 - 2hx + h^2 + D*y^2 - 2Dky + D*k^2 = r^2
        # x^2 + D*y^2 - 2hx - 2Dky + (h^2 + D*k^2 - r^2) = 0
        # x^2 + D*y^2 + Ax + By + C = 0 where A=-2h, B=-2Dk, C=h^2+D*k^2-r^2
        
        # So we need: xi^2 + Di*yi^2 + A*xi + B*yi + C = 0 for all four points
        # This is a system of 4 linear equations in A, B, C (with the constant term being xi^2+Di*yi^2)
        
        # For a non-trivial solution, the determinant must be zero:
        # | x1^2 + D*y1^2  x1  y1  1 |
        # | x2^2 + D*y2^2  x2  y2  1 |
        # | x3^2 + D*y3^2  x3  y3  1 |
        # | x4^2 + D*y4^2  x4  y4  1 |
        
        # Wait, that's not quite right either. Let me reconsider.
        
        # The equation is: x^2 + D*y^2 + A*x + B*y + C = 0
        # For each point (xi, yi): xi^2 + D*yi^2 + A*xi + B*yi + C = 0
        # Rearranging: A*xi + B*yi + C = -(xi^2 + D*yi^2)
        
        # This is a system of linear equations in A, B, C. For 4 points to be concyclic,
        # there must exist A, B, C satisfying all 4 equations. This means the 4x3 system
        # [xi, yi, 1] * [A, B, C]^T = [-(xi^2 + D*yi^2)] must be consistent.
        
        # Equivalently, the augmented matrix must have rank 3 (not 4).
        
        # Let's compute the determinant of the 4x4 matrix:
        # | x1  y1  1  x1^2 + D*y1^2 |
        # | x2  y2  1  x2^2 + D*y2^2 |
        # | x3  y3  1  x3^2 + D*y3^2 |
        # | x4  y4  1  x4^2 + D*y4^2 |
        
        # If this determinant is zero, the points are concyclic (or collinear, but we check that separately)
        
        a11, a12, a13, a14 = x1, y1, Fraction(1), x1*x1 + D*y1*y1
        a21, a22, a23, a24 = x2, y2, Fraction(1), x2*x2 + D*y2*y2
        a31, a32, a33, a34 = x3, y3, Fraction(1), x3*x3 + D*y3*y3
        a41, a42, a43, a44 = x4, y4, Fraction(1), x4*x4 + D*y4*y4
        
        # Compute determinant using cofactor expansion
        det = (a11 * (a22 * (a33 * a44 - a34 * a43) - a23 * (a32 * a44 - a34 * a42) + a24 * (a32 * a43 - a33 * a42))
               - a12 * (a21 * (a33 * a44 - a34 * a43) - a23 * (a31 * a44 - a34 * a41) + a24 * (a31 * a43 - a33 * a41))
               + a13 * (a21 * (a32 * a44 - a34 * a42) - a22 * (a31 * a44 - a34 * a41) + a24 * (a31 * a42 - a32 * a41))
               - a14 * (a21 * (a32 * a43 - a33 * a42) - a22 * (a31 * a43 - a33 * a41) + a23 * (a31 * a42 - a32 * a41)))
        
        return det == 0
    
    def validate(points, D):
        """Full validation: all distances rational, no 3 collinear, no 4 concyclic."""
        n = len(points)
        
        # Check all pairwise distances are rational
        for i in range(n):
            for j in range(i + 1, n):
                if not is_rational_distance(points[i], points[j], D):
                    return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k], D):
                        return False
        
        # Check no four concyclic
        if n >= 4:
            for combo in combinations(range(n), 4):
                if are_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]], D):
                    return False
        
        return True
    
    # Strategy: Use known configurations and try to extend them
    # Known: 5 points with D=1 work: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    
    # Convert base points to our format: (x_num, x_den, y_num, y_den)
    base_points_int = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    base_points = [(x, 1, y, 1) for x, y in base_points_int]
    
    best_points = list(base_points)
    best_D = 1
    
    # Verify base set
    if validate(base_points, 1):
        best_points = list(base_points)
    else:
        # Fallback: use a simple valid set
        best_points = [(0, 1, 0, 1), (1, 1, 0, 1), (0, 1, 1, 1)]
    
    # Try to add points using systematic search
    def try_add_point(points, D, max_coord=100, time_limit=5):
        """Try to find a new point to add."""
        start = time.time()
        point_set = set(points)
        
        # Generate candidate points
        candidates = []
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                if (x, 1, y, 1) not in point_set:
                    candidates.append((x, 1, y, 1))
        
        # Shuffle for randomness
        random.shuffle(candidates)
        
        for candidate in candidates:
            if time.time() - start > time_limit:
                break
            new_points = points + [candidate]
            if validate(new_points, D):
                return new_points
        
        return None
    
    # Try to extend the base set
    if time.time() < deadline - 30:
        result = try_add_point(best_points, 1, 80, 20)
        if result is not None:
            best_points = result
    
    # If we have 6 points, try to add a 7th
    if len(best_points) >= 6 and time.time() < deadline - 20:
        result = try_add_point(best_points, 1, 60, 15)
        if result is not None:
            best_points = result
    
    # If we have 7 points, try to add an 8th
    if len(best_points) >= 7 and time.time() < deadline - 10:
        result = try_add_point(best_points, 1, 50, 8)
        if result is not None:
            best_points = result
    
    # If we have 8 points, try to add a 9th
    if len(best_points) >= 8 and time.time() < deadline - 5:
        result = try_add_point(best_points, 1, 40, 4)
        if result is not None:
            best_points = result
    
    # Format the result
    points_formatted = []
    for (xn, xd, yn, yd) in best_points:
        points_formatted.append([[xn, xd], [yn, yd]])
    
    return {'D': best_D, 'points': points_formatted}
# EVOLVE-BLOCK-END
