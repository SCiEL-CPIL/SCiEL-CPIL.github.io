"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt

def construct():
    """Search for integral point sets with no three collinear and no four concyclic."""
    t_start = time.time()
    DEADLINE = t_start + 145  # 5s safety margin before 150s kill

    def time_ok():
        return time.time() < DEADLINE

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        """Check if three points are collinear using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def check_concyclic(p1, p2, p3, p4):
        """Check if four points are concyclic using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        
        return det == 0

    def validate_incremental(points, new_point):
        """Check if new_point can be added to points (all constraints satisfied)."""
        # Check distance to all existing points
        for p in points:
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_square(d2):
                return False
        
        # Check collinearity with all pairs
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(points[i], points[j], new_point):
                    return False
                if check_collinear(points[i], new_point, points[j]):
                    return False
                if check_collinear(new_point, points[i], points[j]):
                    return False
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        
        return True

    def try_add_point(points, candidate):
        """Try to add a candidate point to the set."""
        if any(candidate == p for p in points):
            return False
        return validate_incremental(points, candidate)

    # Start with known good 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Precompute existing point set for fast membership check
    point_set = set(best_points)
    
    search_range = 150
    random.seed(42)
    
    # Try to add points one at a time, multiple passes
    while time_ok():
        current_size = len(best_points)
        if current_size >= 8:
            break
        
        improved = False
        
        # Strategy 1: Random candidates in expanding box
        candidates = []
        for x in range(-search_range, search_range + 1):
            for y in range(-search_range, search_range + 1):
                if (x, y) not in point_set:
                    candidates.append((x, y))
        
        random.shuffle(candidates)
        
        for cand in candidates:
            if not time_ok():
                break
            if try_add_point(best_points, cand):
                best_points.append(cand)
                point_set.add(cand)
                improved = True
                search_range = min(search_range + 30, 400)
                break
        
        if improved:
            continue
        
        # Strategy 2: Points near existing points (local search)
        for p in best_points:
            if not time_ok():
                break
            for dx in range(-20, 21):
                for dy in range(-20, 21):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (p[0] + dx, p[1] + dy)
                    if cand in point_set:
                        continue
                    if try_add_point(best_points, cand):
                        best_points.append(cand)
                        point_set.add(cand)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Strategy 3: Symmetric extensions
        for p in list(best_points):
            if not time_ok():
                break
            for cand in [(p[0], -p[1]), (-p[0], p[1]), (-p[0], -p[1]),
                         (p[1], p[0]), (p[1], -p[0]), (-p[1], p[0]), (-p[1], -p[0])]:
                if cand in point_set:
                    continue
                if try_add_point(best_points, cand):
                    best_points.append(cand)
                    point_set.add(cand)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Strategy 4: Points on integer circles around existing points
        for p in best_points:
            if not time_ok():
                break
            for r2 in range(1, 200):
                if not is_square(r2):
                    continue
                r = isqrt(r2)
                for dx in range(-r, r+1):
                    dy2 = r2 - dx*dx
                    if dy2 < 0:
                        continue
                    dy = isqrt(dy2)
                    if dy*dy != dy2:
                        continue
                    for dy_sign in [1, -1]:
                        for dx_sign in [1, -1]:
                            cand = (p[0] + dx_sign*dx, p[1] + dy_sign*dy)
                            if cand in point_set:
                                continue
                            if try_add_point(best_points, cand):
                                best_points.append(cand)
                                point_set.add(cand)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a fresh random search with larger range
            if not time_ok():
                break
            search_range += 50
            if search_range > 500:
                break
            # If we've been stuck, try a different starting configuration
            # by removing the least "central" point and trying to rebuild
            if len(best_points) == 5:
                # Try different initial configurations
                alt_configs = [
                    [(0, 0), (3, 4), (0, 5), (-3, 4), (5, 0)],
                    [(0, 0), (6, 8), (15, 8), (15, 0), (9, 12)],
                    [(0, 0), (7, 24), (0, 25), (-7, 24), (15, 20)],
                ]
                for alt in alt_configs:
                    if not time_ok():
                        break
                    # Validate the alternative config
                    valid = True
                    for i in range(len(alt)):
                        for j in range(i+1, len(alt)):
                            dx = alt[i][0] - alt[j][0]
                            dy = alt[i][1] - alt[j][1]
                            if not is_square(dx*dx + dy*dy):
                                valid = False
                                break
                        if not valid:
                            break
                        for k in range(j+1, len(alt)):
                            if check_collinear(alt[i], alt[j], alt[k]):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Check concyclicity
                        for i in range(len(alt)):
                            for j in range(i+1, len(alt)):
                                for k in range(j+1, len(alt)):
                                    for l in range(k+1, len(alt)):
                                        if check_concyclic(alt[i], alt[j], alt[k], alt[l]):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                    if valid and len(alt) >= len(best_points):
                        best_points = alt[:]
                        point_set = set(alt)
                        improved = True
                        break
                if improved:
                    continue
                break
    
    # Format the result
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
