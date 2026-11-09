import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def is_rational_square(frac):
        """Check if a non-negative Fraction is a perfect square of a rational."""
        if frac < 0:
            return False
        if frac == 0:
            return True
        num = frac.numerator
        den = frac.denominator
        # sqrt(num/den) = sqrt(num*den)/den, so num*den must be a perfect square
        prod = num * den
        return is_perfect_square(prod)

    def validate_points(points, D):
        """
        Validate: distinct, rational distances, no 3 collinear, no 4 concyclic.
        Points are (x, y) where geometric coords are (x, y*sqrt(D)).
        D is a positive integer (squarefree).
        """
        n = len(points)
        if n < 3:
            return False

        # Check distinctness
        for i in range(n):
            for j in range(i + 1, n):
                if points[i] == points[j]:
                    return False

        # Check rational distances
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx * dx + D * dy * dy
                if not is_rational_square(d2):
                    return False

        # Check no three collinear
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    # Cross product in geometric space
                    # (x1-x2)*(y3-y1)*sqrt(D) - (x3-x1)*(y2-y1)*sqrt(D)
                    # = sqrt(D) * [(x1-x2)*(y3-y1) - (x3-x1)*(y2-y1)]
                    cross = (x1 - x2) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if cross == 0:
                        return False

        # Check no four concyclic
        # Circle: (x-h)^2 + D*(y-k)^2 = R^2
        # x^2 + D*y^2 + A*x + B*y + C = 0
        # where A = -2h, B = -2D*k, C = h^2 + D*k^2 - R^2
        # For 4 points, solve 3x3 system from first 3, check 4th
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        # Solve A, B, C from first 3 points
                        # x1^2 + D*y1^2 + A*x1 + B*y1 + C = 0
                        a1 = p1[0] * p1[0] + D * p1[1] * p1[1]
                        a2 = p2[0] * p2[0] + D * p2[1] * p2[1]
                        a3 = p3[0] * p3[0] + D * p3[1] * p3[1]

                        # System:
                        # x1*A + y1*B + C = -a1
                        # x2*A + y2*B + C = -a2
                        # x3*A + y3*B + C = -a3

                        # Subtract row1 from row2 and row3:
                        # (x2-x1)*A + (y2-y1)*B = -(a2-a1)
                        # (x3-x1)*A + (y3-y1)*B = -(a3-a1)

                        dx1 = p2[0] - p1[0]
                        dy1 = p2[1] - p1[1]
                        dx2 = p3[0] - p1[0]
                        dy2 = p3[1] - p1[1]
                        c1 = -(a2 - a1)
                        c2 = -(a3 - a1)

                        det = dx1 * dy2 - dx2 * dy1
                        if det == 0:
                            # Should not happen (no 3 collinear)
                            continue

                        # A = (c1*dy2 - c2*dy1) / det
                        # B = (dx1*c2 - dx2*c1) / det
                        # C = -a1 - A*x1 - B*y1

                        A_num = c1 * dy2 - c2 * dy1
                        B_num = dx1 * c2 - dx2 * c1

                        # Check 4th point: x4^2 + D*y4^2 + A*x4 + B*y4 + C = 0
                        # Substitute C = -a1 - A*x1 - B*y1:
                        # x4^2 + D*y4^2 + A*x4 + B*y4 - a1 - A*x1 - B*y1 = 0
                        # (x4^2 + D*y4^2 - a1) + A*(x4-x1) + B*(y4-y1) = 0
                        # Multiply by det:
                        # det*(x4^2 + D*y4^2 - a1) + A_num*(x4-x1) + B_num*(y4-y1) = 0

                        a4 = p4[0] * p4[0] + D * p4[1] * p4[1]
                        val = det * (a4 - a1) + A_num * (p4[0] - p1[0]) + B_num * (p4[1] - p1[1])
                        if val == 0:
                            # They are concyclic - check if it's a real circle (R^2 > 0)
                            # R^2 = h^2 + D*k^2 - C
                            # h = -A/2, k = -B/(2D)
                            # R^2 = A^2/4 + B^2/(4D) - C
                            # C = -a1 - A*x1 - B*y1
                            # R^2 = A^2/4 + B^2/(4D) + a1 + A*x1 + B*y1
                            # With A = A_num/det, B = B_num/det:
                            # R^2 = A_num^2/(4*det^2) + B_num^2/(4*D*det^2) + a1 + A_num*x1/det + B_num*y1/det
                            # Multiply by 4*det^2:
                            # 4*det^2*R^2 = A_num^2 + B_num^2/D + 4*a1*det^2 + 4*A_num*x1*det + 4*B_num*y1*det
                            # For R^2 > 0, we need this > 0
                            # Since D > 0 and det^2 > 0, check:
                            r2_num = A_num * A_num + B_num * B_num // D if D == 1 else None
                            # Simpler: just check if the "circle" is degenerate
                            # If R^2 <= 0, it's not a valid circle, so not concyclic in the geometric sense
                            # For our purposes, if val==0 and the points form a valid circle, return False
                            # A valid circle requires R^2 > 0
                            # Let's compute R^2 exactly
                            A_frac = Fraction(A_num, det)
                            B_frac = Fraction(B_num, det)
                            C_frac = -Fraction(a1) - A_frac * p1[0] - B_frac * p1[1]
                            R2 = A_frac * A_frac / 4 + B_frac * B_frac / (4 * D) - C_frac
                            if R2 > 0:
                                return False
        return True

    # Known 5-point solution
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1

    if not validate_points(points_5, D):
        # Fallback
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    best_points = points_5[:]
    best_count = 5

    # Try to find a 6th point by searching
    # Search in a reasonable range
    # The existing points span x: [-56, 16], y: [0, 63]
    # Try extending the search

    # First, try some known candidate positions
    # For integral point sets, common patterns involve Pythagorean triples
    # Try points that maintain rational distances to all 5 existing points

    search_range = range(-100, 101)

    for x in search_range:
        if time.time() > deadline:
            break
        for y in search_range:
            if time.time() > deadline:
                break
            new_point = (x, y)
            if new_point in best_points:
                continue

            # Quick check: rational distances to all existing points
            valid_dist = True
            for px, py in best_points:
                dx = x - px
                dy = y - py
                d2 = dx * dx + dy * dy
                if not is_perfect_square(d2):
                    valid_dist = False
                    break

            if not valid_dist:
                continue

            candidate = best_points[:] + [new_point]
            if validate_points(candidate, D):
                best_points = candidate
                best_count += 1
                # Try to find another point
                if best_count >= 7:
                    break
        if best_count >= 7:
            break

    # Try to find more points if we found a 6th
    if best_count == 6:
        for x in range(-150, 151):
            if time.time() > deadline:
                break
            for y in range(-150, 151):
                if time.time() > deadline:
                    break
                new_point = (x, y)
                if new_point in best_points:
                    continue

                valid_dist = True
                for px, py in best_points:
                    dx = x - px
                    dy = y - py
                    d2 = dx * dx + dy * dy
                    if not is_perfect_square(d2):
                        valid_dist = False
                        break

                if not valid_dist:
                    continue

                candidate = best_points[:] + [new_point]
                if validate_points(candidate, D):
                    best_points = candidate
                    best_count += 1
                    break
            if best_count >= 7 or time.time() > deadline:
                break

    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
