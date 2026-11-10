"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt
from itertools import combinations

def construct():
    """Search for integral point sets with no three collinear and no four concyclic."""
    t_start = time.time()
    DEADLINE = t_start + 140  # 10s safety margin before 150s kill

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
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        
        return True

    def try_add_point(points, point_set, candidate):
        """Try to add a candidate point to the set."""
        if candidate in point_set:
            return False
        return validate_incremental(points, candidate)

    def generate_pythagorean_candidates(center, max_radius=300):
        """Generate points at integer distance from center."""
        candidates = set()
        for r2 in range(1, max_radius*max_radius + 1):
            r = isqrt(r2)
            if r*r != r2:
                continue
            for dx in range(-r, r+1):
                dy2 = r2 - dx*dx
                if dy2 < 0:
                    continue
                dy = isqrt(dy2)
                if dy*dy != dy2:
                    continue
                for dy_sign in ([1, -1] if dy != 0 else [0]):
                    for dx_sign in ([1, -1] if dx != 0 else [0]):
                        cand = (center[0] + dx_sign*dx, center[1] + dy_sign*dy)
                        if cand != center:
                            candidates.add(cand)
        return candidates

    def find_best_candidate(points, point_set, candidates):
        """Find the best candidate to add from a list."""
        for cand in candidates:
            if cand in point_set:
                continue
            if validate_incremental(points, cand):
                return cand
        return None

    # Start with known good 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    point_set = set(best_points)
    
    random.seed(42)
    search_range = 150
    stuck_counter = 0
    max_stuck = 5
    
    # Try to add points one at a time, multiple passes
    while time_ok():
        current_size = len(best_points)
        if current_size >= 8:
            break
        
        improved = False
        
        # Strategy 1: Generate candidates using Pythagorean constraints from existing points
        if not time_ok():
            break
        
        # Collect all Pythagorean candidates from each existing point
        all_candidates = set()
        for p in best_points:
            pyth_cands = generate_pythagorean_candidates(p, max_radius=200)
            all_candidates.update(pyth_cands)
        
        # Also add some random candidates
        for _ in range(500):
            x = random.randint(-search_range, search_range)
            y = random.randint(-search_range, search_range)
            all_candidates.add((x, y))
        
        # Remove existing points
        all_candidates -= point_set
        
        # Convert to list and shuffle for diversity
        candidate_list = list(all_candidates)
        random.shuffle(candidate_list)
        
        # Try to find a valid candidate
        found = find_best_candidate(best_points, point_set, candidate_list)
        if found:
            best_points.append(found)
            point_set.add(found)
            improved = True
            stuck_counter = 0
            search_range = min(search_range + 20, 400)
        
        if improved:
            continue
        
        # Strategy 2: Local search - points near existing points
        for p in best_points:
            if not time_ok():
                break
            local_cands = []
            for dx in range(-30, 31):
                for dy in range(-30, 31):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (p[0] + dx, p[1] + dy)
                    if cand not in point_set:
                        local_cands.append(cand)
            
            random.shuffle(local_cands)
            found = find_best_candidate(best_points, point_set, local_cands)
            if found:
                best_points.append(found)
                point_set.add(found)
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 3: Symmetric extensions
        for p in list(best_points):
            if not time_ok():
                break
            sym_cands = [(p[0], -p[1]), (-p[0], p[1]), (-p[0], -p[1]),
                         (p[1], p[0]), (p[1], -p[0]), (-p[1], p[0]), (-p[1], -p[0])]
            sym_cands = [c for c in sym_cands if c not in point_set]
            found = find_best_candidate(best_points, point_set, sym_cands)
            if found:
                best_points.append(found)
                point_set.add(found)
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 4: Backtracking - remove a point and try different additions
        stuck_counter += 1
        if stuck_counter >= max_stuck and len(best_points) > 3:
            # Remove the last added point (or a random point)
            idx = len(best_points) - 1 if random.random() < 0.5 else random.randint(0, len(best_points)-1)
            removed = best_points.pop(idx)
            point_set.discard(removed)
            stuck_counter = 0
            # Try to add a different point
            all_candidates = set()
            for p in best_points:
                pyth_cands = generate_pythagorean_candidates(p, max_radius=250)
                all_candidates.update(pyth_cands)
            all_candidates -= point_set
            candidate_list = list(all_candidates)
            random.shuffle(candidate_list)
            found = find_best_candidate(best_points, point_set, candidate_list)
            if found:
                best_points.append(found)
                point_set.add(found)
            else:
                # Restore the removed point
                best_points.append(removed)
                point_set.add(removed)
        
        # If still stuck, expand search range
        if not improved and stuck_counter < max_stuck:
            search_range += 30
            if search_range > 500:
                break
    
    # Final validation
    valid = True
    for i in range(len(best_points)):
        for j in range(i+1, len(best_points)):
            dx = best_points[i][0] - best_points[j][0]
            dy = best_points[i][1] - best_points[j][1]
            if not is_square(dx*dx + dy*dy):
                valid = False
                break
        if not valid:
            break
        for k in range(j+1, len(best_points)):
            if check_collinear(best_points[i], best_points[j], best_points[k]):
                valid = False
                break
        if not valid:
            break
    if valid:
        for i in range(len(best_points)):
            for j in range(i+1, len(best_points)):
                for k in range(j+1, len(best_points)):
                    for l in range(k+1, len(best_points)):
                        if check_concyclic(best_points[i], best_points[j], best_points[k], best_points[l]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback to known good set
            best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Format the result
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
