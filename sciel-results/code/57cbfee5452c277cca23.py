"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
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

    def generate_candidates_from_point(p, D, max_radius=200):
        """Generate candidate points at rational distance from p using Pythagorean-like structure."""
        candidates = set()
        x0, y0 = p
        # For D=1: (x-x0)^2 + (y-y0)^2 = r^2 where r is integer
        # We can parameterize: x = x0 + r*cos(t), y = y0 + r*sin(t)
        # For integer solutions, use Pythagorean triples
        for r in range(1, max_radius + 1):
            if time.time() > deadline - 5:
                break
            # Generate all integer points on circle of radius r
            for dx in range(-r, r + 1):
                dy2 = r * r - dx * dx
                if dy2 < 0:
                    continue
                dy = math.isqrt(dy2)
                if dy * dy == dy2:
                    candidates.add((x0 + dx, y0 + dy))
                    if dy != 0:
                        candidates.add((x0 + dx, y0 - dy))
        return candidates

    def try_add_point_smart(current_points, D=1, max_radius=150):
        """Try to find a new point using smart candidate generation."""
        existing = set(current_points)
        candidates = set()
        
        # Generate candidates from each existing point
        for p in current_points:
            if time.time() > deadline - 5:
                return None
            cand = generate_candidates_from_point(p, D, max_radius)
            candidates.update(cand)
        
        # Filter out existing points
        candidates = [c for c in candidates if c not in existing]
        
        # Sort candidates by some heuristic (e.g., distance from origin)
        candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
        
        # Try each candidate
        for p in candidates:
            if time.time() > deadline - 5:
                return None
            if validate_new_point(current_points, p, D):
                return current_points + [p]
        
        return None

    def try_add_point_grid(current_points, D=1, max_coord=100):
        """Fallback: try to find a new point using grid search."""
        existing = set(current_points)
        
        for x in range(-max_coord, max_coord + 1):
            if time.time() > deadline - 5:
                return None
            for y in range(-max_coord, max_coord + 1):
                p = (x, y)
                if p in existing:
                    continue
                if validate_new_point(current_points, p, D):
                    return current_points + [p]
        return None

    # Try multiple base configurations
    base_configs = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],  # Known 5-point
        [(0, 0), (3, 4), (0, 5), (-4, 3), (4, 3)],          # Another 5-point candidate
        [(0, 0), (5, 0), (0, 5), (3, 4), (4, 3)],           # Another candidate
    ]
    
    best_set = []
    best_D = 1
    
    for base in base_configs:
        if time.time() > deadline - 20:
            break
        
        # Verify base configuration
        valid = True
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                if not check_dist_rational(base[i], base[j], 1):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check no three collinear in base
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                for k in range(j + 1, len(base)):
                    if check_collinear(base[i], base[j], base[k]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        current = list(base)
        
        # Try to add points
        for _ in range(7):  # Try to add up to 7 more points
            if time.time() > deadline - 10:
                break
            
            # First try smart search
            result = try_add_point_smart(current, 1, 120)
            if result is None and time.time() < deadline - 10:
                # Fallback to grid search with smaller range
                result = try_add_point_grid(current, 1, 80)
            
            if result is None:
                break
            
            current = result
        
        if len(current) > len(best_set):
            best_set = current
            best_D = 1
    
    # If we didn't find anything better, use the original 5-point set
    if len(best_set) < 5:
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_D = 1
    
    # Format the result
    points_formatted = []
    for x, y in best_set:
        points_formatted.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': points_formatted}
# EVOLVE-BLOCK-END
