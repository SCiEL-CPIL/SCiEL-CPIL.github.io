"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    def check_dist_rational(p1, p2, D=1):
        """Check if distance between p1 and p2 is rational."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx * dx + D * dy * dy
        return is_square(d2)

    def check_collinear(p1, p2, p3):
        """Check if three points are collinear."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0

    def check_concyclic(p1, p2, p3, p4):
        """Check if four points are concyclic."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        a1 = x1 * x1 + y1 * y1
        a2 = x2 * x2 + y2 * y2
        a3 = x3 * x3 + y3 * y3
        a4 = x4 * x4 + y4 * y4
        
        det = (
            a1 * (x2 * (y3 - y4) - y2 * (x3 - x4) + (x3 * y4 - x4 * y3))
            - x1 * (a2 * (y3 - y4) - y2 * (a3 - a4) + (a3 * y4 - a4 * y3))
            + y1 * (a2 * (x3 - x4) - x2 * (a3 - a4) + (a3 * x4 - a4 * x3))
            - 1 * (a2 * (x3 * y4 - x4 * y3) - x2 * (a3 * y4 - a4 * y3) + y2 * (a3 * x4 - a4 * x3))
        )
        return det == 0

    def validate_new_point(current_points, new_point, D=1):
        """Check if adding new_point to current_points maintains all properties."""
        # 1. Check distances to all existing points
        for p in current_points:
            if not check_dist_rational(new_point, p, D):
                return False
        
        # 2. Check collinearity with any pair of existing points
        n = len(current_points)
        for i in range(n):
            for j in range(i + 1, n):
                if check_collinear(new_point, current_points[i], current_points[j]):
                    return False
        
        # 3. Check concyclicity with any triple of existing points
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if check_concyclic(new_point, current_points[i], current_points[j], current_points[k]):
                        return False
        
        return True

    def try_add_point(current_points, D=1, max_coord=500):
        """Try to find a new point to add to current_points using systematic search."""
        n = len(current_points)
        # Use a set for O(1) lookup of existing points
        existing = set(current_points)
        
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                if time.time() > deadline - 5:
                    return None
                
                p = (x, y)
                if p in existing:
                    continue
                
                if validate_new_point(current_points, p, D):
                    return current_points + [p]
        return None

    # Verify the base set is valid (it is known to be valid)
    # best_set = list(base_points)
    
    # Try to add a 6th point using systematic search with a larger range
    if time.time() < deadline - 10:
        result = try_add_point(best_set, 1, 300)
        if result is not None:
            best_set = result
    
    # If we found 6 points, try to find a 7th
    if len(best_set) >= 6 and time.time() < deadline - 10:
        result = try_add_point(best_set, 1, 200)
        if result is not None:
            best_set = result
            
    # If we found 7 points, try to find an 8th
    if len(best_set) >= 7 and time.time() < deadline - 5:
        result = try_add_point(best_set, 1, 100)
        if result is not None:
            best_set = result

    # Format the result
    points_formatted = []
    for x, y in best_set:
        points_formatted.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': points_formatted}
# EVOLVE-BLOCK-END
