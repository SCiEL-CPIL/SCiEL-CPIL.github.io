import random
import math
from itertools import combinations

def construct():
    # Box size for search
    BOX = 25
    
    def get_normalized_slope(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return (0, 0) # Should not happen for distinct points
        # Normalize direction. We need to distinguish (1,1) from (-1,-1)? 
        # No, for collinearity, (1,1) and (-1,-1) are the same line direction.
        # But (1,1) and (1,-1) are different.
        # So we can just divide by GCD and ensure a canonical sign.
        # e.g. always make dx > 0, or if dx == 0, dy > 0.
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        return (dx, dy)

    def score(points_list):
        # Returns number of lines with exactly 4 points, or -1 if invalid (5 collinear or duplicates)
        n = len(points_list)
        # Check distinctness
        pts_set = set(tuple(p) for p in points_list)
        if len(pts_set) != n:
            return -1
        
        total_lines_4 = 0
        for i in range(n):
            p_i = points_list[i]
            slopes = {}
            for j in range(n):
                if i == j:
                    continue
                p_j = points_list[j]
                s = get_normalized_slope(p_i, p_j)
                slopes[s] = slopes.get(s, 0) + 1
            
            for s, count in slopes.items():
                if count >= 4:
                    return -1 # 5 or more collinear
                elif count == 3:
                    total_lines_4 += 1
        
        # Each line of 4 points is counted 4 times (once for each point on the line)
        return total_lines_4 // 4

    # Initialize points randomly in the box
    def random_point():
        return [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
    
    points = [random_point() for _ in range(16)]
    current_score = score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    iterations = 20000 # Adjust based on time limit
    
    for it in range(iterations):
        if T < T_min:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_point = points[idx]
        new_point = random_point()
        
        # Ensure distinctness quickly
        if new_point in points and points.index(new_point) != idx:
            continue
        # Actually, `new_point in points` is O(N). 
        # Better to just swap and check score? 
        # The score function checks distinctness. 
        # But generating duplicates wastes time. 
        # Let's just generate and let score handle it, or check set.
        
        # Optimization: Check if new_point collides with others
        # This is O(N). Score is O(N^2). 
        # Let's just do the swap and score.
        
        points[idx] = new_point
        new_score = score(points)
        
        if new_score > current_score:
            current_score = new_score
            points = [p[:] for p in points]
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            # Accept with probability exp((new_score - current_score) / T)
            if new_score == -1 or current_score == -1:
                # If we went from valid to invalid, reject (or accept with very low prob)
                # If we went from invalid to valid, accept
                if new_score > current_score:
                     # Already handled above
                     pass
                else:
                     # Reject invalid moves usually, unless stuck?
                     # Let's just revert
                     points[idx] = old_point
            else:
                delta = new_score - current_score
                # delta is negative
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    points[idx] = old_point
        
        T *= cooling_rate
        
        # Optional: Random restart if stuck too long?
        # For now, just SA.

    return [[int(x), int(y)] for x, y in best_points]
