import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Start with the known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = list(base_points)
    best_D = 1

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2, D=1):
        """Return squared distance as a rational number (Fraction)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + D * dy * dy

    def all_dist_rational(points, D=1):
        """Check if all pairwise distances are rational."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq(points[i], points[j], D)
                # d2 must be a perfect square rational
                num = d2.numerator
                den = d2.denominator
                if not is_square(num) or not is_square(den):
                    return False
        return True

    def no_three_collinear(points):
        """Check no three points are collinear using exact arithmetic."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Determinant of the 3x3 matrix
                    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if det == 0:
                        return False
        return True

    def no_four_concyclic(points):
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
                        # Four points are concyclic iff the determinant of the 4x4 matrix is 0
                        # | x^2+y^2  x  y  1 |
                        # | x1^2+y1^2 x1 y1 1 |
                        # | x2^2+y2^2 x2 y2 1 |
                        # | x3^2+y3^2 x3 y3 1 |
                        # | x4^2+y4^2 x4 y4 1 |
                        # Actually, we need a 4x4 determinant. Let's use the standard test.
                        # A circle has equation x^2 + y^2 + Dx + Ey + F = 0
                        # For 4 points, the system must be consistent.
                        # We can check if the determinant of the 4x4 matrix is zero:
                        # | x1^2+y1^2  x1  y1  1 |
                        # | x2^2+y2^2  x2  y2  1 |
                        # | x3^2+y3^2  x3  y3  1 |
                        # | x4^2+y4^2  x4  y4  1 |
                        a1 = x1 * x1 + y1 * y1
                        a2 = x2 * x2 + y2 * y2
                        a3 = x3 * x3 + y3 * y3
                        a4 = x4 * x4 + y4 * y4
                        # Compute determinant of 4x4 matrix
                        # M = [[a1,x1,y1,1],[a2,x2,y2,1],[a3,x3,y3,1],[a4,x4,y4,1]]
                        # Use expansion or direct formula. Let's use a simple 4x4 determinant.
                        # det = a1*(x2*(y3*1 - y4*1) - y2*(x3*1 - x4*1) + 1*(x3*y4 - x4*y3))
                        #     - x1*(a2*(y3*1 - y4*1) - y2*(a3*1 - a4*1) + 1*(a3*y4 - a4*y3))
                        #     + y1*(a2*(x3*1 - x4*1) - x2*(a3*1 - a4*1) + 1*(a3*x4 - a4*x3))
                        #     - 1*(a2*(x3*y4 - x4*y3) - x2*(a3*y4 - a4*y3) + y2*(a3*x4 - a4*x3))
                        det = (
                            a1 * (x2 * (y3 - y4) - y2 * (x3 - x4) + (x3 * y4 - x4 * y3))
                            - x1 * (a2 * (y3 - y4) - y2 * (a3 - a4) + (a3 * y4 - a4 * y3))
                            + y1 * (a2 * (x3 - x4) - x2 * (a3 - a4) + (a3 * x4 - a4 * x3))
                            - 1 * (a2 * (x3 * y4 - x4 * y3) - x2 * (a3 * y4 - a4 * y3) + y2 * (a3 * x4 - a4 * x3))
                        )
                        if det == 0:
                            return False
        return True

    def validate(points, D=1):
        """Full validation: all distances rational, no 3 collinear, no 4 concyclic."""
        if not all_dist_rational(points, D):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True

    # Verify the base set is valid
    if validate(base_points, 1):
        best_set = list(base_points)

    # Try to add a 6th point by searching in a reasonable range
    # We'll search for integer points (x, y) with |x|, |y| <= max_coord
    # that maintain all conditions

    def try_add_point(current_points, D=1, max_coord=200):
        """Try to find a new point to add to current_points."""
        n = len(current_points)
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                # Skip if point already exists
                if (x, y) in current_points:
                    continue
                new_points = current_points + [(x, y)]
                if validate(new_points, D):
                    return new_points
        return None

    # Try to add a 6th point
    if time.time() < deadline - 10:
        result = try_add_point(best_set, 1, 150)
        if result is not None:
            best_set = result

    # If we found 6 points, try to find a 7th
    if len(best_set) >= 6 and time.time() < deadline - 5:
        result = try_add_point(best_set, 1, 100)
        if result is not None:
            best_set = result

    # Format the result
    points_formatted = []
    for x, y in best_set:
        points_formatted.append([[x, 1], [y, 1]])

    return {'D': best_D, 'points': points_formatted}
