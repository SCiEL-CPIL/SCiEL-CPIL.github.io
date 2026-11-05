"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Known 5-point configuration with all integer distances
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = list(base_points)
    best_D = 1
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    def dist_sq_int(p1, p2, D=1):
        """Return squared distance as integer (coords are integers)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + D * dy * dy
    
    def all_dist_rational(points, D=1):
        """Check if all pairwise distances are rational (integer squared for integer coords)."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq_int(points[i], points[j], D)
                if not is_square(d2):
                    return False
        return True
    
    def no_three_collinear(points):
        """Check no three points are collinear using exact integer arithmetic."""
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx12 = x2 - x1
                dy12 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    if dx12 * (y3 - y1) - (x3 - x1) * dy12 == 0:
                        return False
        return True
    
    def no_four_concyclic(points, D=1):
        """Check no four points are concyclic using exact integer arithmetic."""
        n = len(points)
        for quad in combinations(range(n), 4):
            i, j, k, l = quad
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            x4, y4 = points[l]
            a1 = x1*x1 + D*y1*y1
            a2 = x2*x2 + D*y2*y2
            a3 = x3*x3 + D*y3*y3
            a4 = x4*x4 + D*y4*y4
            det = (
                a1 * (x2*(y3-y4) - y2*(x3-x4) + (x3*y4 - x4*y3))
                - x1 * (a2*(y3-y4) - y2*(a3-a4) + (a3*y4 - a4*y3))
                + y1 * (a2*(x3-x4) - x2*(a3-a4) + (a3*x4 - a4*x3))
                - 1 * (a2*(x3*y4 - x4*y3) - x2*(a3*y4 - a4*y3) + y2*(a3*x4 - a4*x3))
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
        if not no_four_concyclic(points, D):
            return False
        return True
    
    # Verify the base set is valid
    if validate(base_points, 1):
        best_set = list(base_points)
    
    def try_add_point_fast(current_points, D=1, max_coord=300):
        """Try to find a new point to add using targeted search with fast fail."""
        point_set = set(current_points)
        n = len(current_points)
        
        for x in range(-max_coord, max_coord + 1):
            if time.time() > deadline - 5:
                return None
            for y in range(-max_coord, max_coord + 1):
                if (x, y) in point_set:
                    continue
                new_pt = (x, y)
                # Check all distances first (fastest to fail)
                valid = True
                for pt in current_points:
                    d2 = dist_sq_int(new_pt, pt, D)
                    if not is_square(d2):
                        valid = False
                        break
                if not valid:
                    continue
                # Check collinearity
                collinear = False
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1 = current_points[i]
                        x2, y2 = current_points[j]
                        if (x2-x1)*(y-y1) - (x-x1)*(y2-y1) == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                # Check concyclicity (most expensive)
                new_points = current_points + [new_pt]
                if no_four_concyclic(new_points, D):
                    return new_points
        return None
    
    def randomized_search_fast(current_points, D=1, num_trials=100000):
        """Randomized search for a new point with fast validation."""
        point_set = set(current_points)
        n = len(current_points)
        max_x = max(abs(p[0]) for p in current_points) + 200
        max_y = max(abs(p[1]) for p in current_points) + 200
        
        for _ in range(num_trials):
            if time.time() > deadline - 5:
                return None
            x = random.randint(-max_x, max_x)
            y = random.randint(-max_y, max_y)
            if (x, y) in point_set:
                continue
            new_pt = (x, y)
            # Check all distances first
            valid = True
            for pt in current_points:
                d2 = dist_sq_int(new_pt, pt, D)
                if not is_square(d2):
                    valid = False
                    break
            if not valid:
                continue
            # Check collinearity
            collinear = False
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1 = current_points[i]
                    x2, y2 = current_points[j]
                    if (x2-x1)*(y-y1) - (x-x1)*(y2-y1) == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            # Check concyclicity
            new_points = current_points + [new_pt]
            if no_four_concyclic(new_points, D):
                return new_points
        return None
    
    # Try to add a 6th point
    if time.time() < deadline - 30:
        result = try_add_point_fast(best_set, 1, 150)
        if result is not None:
            best_set = result
    
    if len(best_set) == 5 and time.time() < deadline - 15:
        result = randomized_search_fast(best_set, 1, 50000)
        if result is not None:
            best_set = result
    
    # If we found 6 points, try to find a 7th
    if len(best_set) >= 6 and time.time() < deadline - 10:
        result = try_add_point_fast(best_set, 1, 100)
        if result is not None:
            best_set = result
    if len(best_set) == 6 and time.time() < deadline - 5:
        result = randomized_search_fast(best_set, 1, 20000)
        if result is not None:
            best_set = result
    
    # If we found 7 points, try to find an 8th
    if len(best_set) >= 7 and time.time() < deadline - 5:
        result = try_add_point_fast(best_set, 1, 80)
        if result is not None:
            best_set = result
    if len(best_set) == 7 and time.time() < deadline - 3:
        result = randomized_search_fast(best_set, 1, 10000)
        if result is not None:
            best_set = result
    
    # Format the result
    points_formatted = []
    for x, y in best_set:
        points_formatted.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': points_formatted}
# EVOLVE-BLOCK-END
