import time
import random
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Initial configuration: 4x4 grid
    # This has score 10.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def evaluate(points):
        n = len(points)
        line_counts = {}
        
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    return -10000 # Distinctness check
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * xi - dx * yi
                key = (dx, dy, c)
                line_counts[key] = line_counts.get(key, 0) + 1
        
        score = 0
        invalid = False
        for count in line_counts.values():
            if count == 6: # 4 points -> C(4,2)=6 pairs
                score += 1
            elif count >= 10: # 5 points -> C(5,2)=10 pairs
                invalid = True
                break
        
        if invalid:
            return -1000 + score
        return score

    def evaluate_fast(points):
        """Faster evaluation using numpy for line counting"""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        
        # Generate all pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        p1 = pts[i_idx]
        p2 = pts[j_idx]
        
        dx = p2[:, 0] - p1[:, 0]
        dy = p2[:, 1] - p1[:, 1]
        
        # Check for duplicates
        if np.any((dx == 0) & (dy == 0)):
            return -10000
        
        # Compute gcd
        g = np.zeros(len(dx), dtype=np.int64)
        for k in range(len(dx)):
            g[k] = math.gcd(abs(int(dx[k])), abs(int(dy[k])))
        
        dx_norm = dx // g
        dy_norm = dy // g
        
        # Normalize direction
        flip = (dx_norm < 0) | ((dx_norm == 0) & (dy_norm < 0))
        dx_norm[flip] = -dx_norm[flip]
        dy_norm[flip] = -dy_norm[flip]
        
        c = dy_norm * p1[:, 0] - dx_norm * p1[:, 1]
        
        # Create unique keys
        keys = list(zip(dx_norm.tolist(), dy_norm.tolist(), c.tolist()))
        
        # Count occurrences
        line_counts = {}
        for key in keys:
            line_counts[key] = line_counts.get(key, 0) + 1
        
        score = 0
        invalid = False
        for count in line_counts.values():
            if count == 6: # 4 points -> C(4,2)=6 pairs
                score += 1
            elif count >= 10: # 5 points -> C(5,2)=10 pairs
                invalid = True
                break
        
        if invalid:
            return -1000 + score
        return score

    best_points = [p[:] for p in initial_points]
    best_score = evaluate(best_points)
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Local Search with Random Restarts
    # We will use a simple hill climbing with random kicks.
    
    def random_move(points):
        idx = random.randint(0, 15)
        # Try a small move first
        if random.random() < 0.5:
            new_x = points[idx][0] + random.randint(-5, 5)
            new_y = points[idx][1] + random.randint(-5, 5)
        else:
            # Larger move
            new_x = points[idx][0] + random.randint(-20, 20)
            new_y = points[idx][1] + random.randint(-20, 20)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None
            
        new_point = [new_x, new_y]
        # Check distinctness
        for k in range(16):
            if k != idx and points[k] == new_point:
                return None
        return idx, new_point

    def targeted_move(points, current_score):
        """Try to create a 4-point line by moving one point to align with 3 others"""
        n = len(points)
        
        # Find lines with exactly 3 points
        line_to_points = {}
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)
        
        # Find lines with exactly 3 points
        for key, point_set in line_to_points.items():
            if len(point_set) == 3:
                # Try to find a position for the 4th point
                # Get the 3 points
                pts = [points[i] for i in point_set]
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                x3, y3 = pts[2]
                
                # The 4th point should be on the line through these 3 points
                # For integer points, we need to find a point on the line
                # that is not already one of the 3 points
                
                # Compute direction vector
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                
                # Try positions along the line
                for t in range(-20, 21):
                    new_x = x1 + t * dx
                    new_y = y1 + t * dy
                    
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    
                    new_point = [new_x, new_y]
                    
                    # Check if it's one of the existing 3 points
                    if new_point == pts[0] or new_point == pts[1] or new_point == pts[2]:
                        continue
                    
                    # Check distinctness from all other points
                    is_distinct = True
                    for k in range(n):
                        if points[k] == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                    
                    # Check that this doesn't create a 5-point line
                    # Count how many other points are on this line
                    line_count = 0
                    for k in range(n):
                        if k in point_set:
                            continue
                        # Check if points[k] is on the line
                        px, py = points[k]
                        # Line equation: (y - y1) * dx = (x - x1) * dy
                        if (py - y1) * dx == (px - x1) * dy:
                            line_count += 1
                    
                    if line_count >= 2:  # Would make 5+ points on line
                        continue
                    
                    # Choose which of the 3 points to move
                    idx = random.choice(list(point_set))
                    return idx, new_point
        
        return None

    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Try targeted move first (more likely to improve)
        if random.random() < 0.3:
            move = targeted_move(current_points, current_score)
        else:
            move = random_move(current_points)
        
        if move is None:
            continue
        
        idx, new_point = move
        old_point = current_points[idx][:]
        current_points[idx] = new_point
        new_score = evaluate_fast(current_points)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[idx] = old_point
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Kick: move a random point to a random location in a larger range
            # to escape local optimum
            idx = random.randint(0, 15)
            new_x = random.randint(-200, 200)
            new_y = random.randint(-200, 200)
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                # Check distinctness
                is_distinct = True
                for k in range(16):
                    if k != idx and current_points[k] == [new_x, new_y]:
                        is_distinct = False
                        break
                if is_distinct:
                    current_points[idx] = [new_x, new_y]
                    current_score = evaluate_fast(current_points)
            no_improve_count = 0

    return best_points
