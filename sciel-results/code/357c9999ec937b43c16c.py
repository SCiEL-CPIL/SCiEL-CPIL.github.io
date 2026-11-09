# EVOLVE-BLOCK-START
import math
import random
import time

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def is_valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in points):
        return False
    # Check not all collinear
    (x0, y0) = points[0]
    (x1, y1) = points[1]
    (x2, y2) = points[2]
    if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
        all_collinear = True
        for i in range(3, 12):
            (x3, y3) = points[i]
            if (x1 - x0) * (y3 - y0) != (y1 - y0) * (x3 - x0):
                all_collinear = False
                break
        if all_collinear:
            return False
    return True

def construct():
    best_points = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 110
    
    # Start with a 3x4 grid as a reasonable initial structure
    initial_points = [[x, y] for x in range(4) for y in range(3)]
    
    for restart in range(50):
        # Perturb the initial points
        points = []
        for p in initial_points:
            x = p[0] + random.randint(-2, 2)
            y = p[1] + random.randint(-2, 2)
            points.append([x, y])
        
        # Ensure distinctness
        # If collision, jitter
        for i in range(12):
            if i > 0:
                while points[i] in points[:i]:
                    points[i][0] += random.randint(-1, 1)
                    points[i][1] += random.randint(-1, 1)
        
        if not is_valid(points):
            # Fallback to valid random
            points = [[i, 0] for i in range(11)] + [[0, 1]]
            if not is_valid(points):
                 points = [[i, i] for i in range(11)] + [[0, 1]]
        
        current_score = count_ordinary_lines(points)
        current_points = [p[:] for p in points]
        
        # Local Search with Line Snapping
        T = 1.0
        T_min = 0.01
        alpha = 0.99
        
        while T > T_min and time.time() - start_time < time_limit:
            improved = False
            
            # Strategy 1: Line Snapping
            if random.random() < 0.5:
                idx = random.randint(0, 11)
                others = [i for i in range(12) if i != idx]
                a_idx, b_idx = random.sample(others, 2)
                
                ax, ay = current_points[a_idx]
                bx, by = current_points[b_idx]
                
                dx = bx - ax
                dy = by - ay
                
                # Line equation: dy*x - dx*y = c
                c = dy * ax - dx * ay
                
                # Find closest integer point on line to current_points[idx]
                px, py = current_points[idx]
                
                # Projection of P onto line AB
                # Vector AB = (dx, dy)
                # Vector AP = (px - ax, py - ay)
                # t = dot(AP, AB) / dot(AB, AB)
                dot_ap_ab = (px - ax) * dx + (py - ay) * dy
                dot_ab_ab = dx * dx + dy * dy
                
                if dot_ab_ab == 0:
                    continue
                
                t = dot_ap_ab / dot_ab_ab
                proj_x = ax + t * dx
                proj_y = ay + t * dy
                
                # Round to nearest integer
                new_x = round(proj_x)
                new_y = round(proj_y)
                
                # Check if this point is actually on the line
                if dy * new_x - dx * new_y != c:
                    # Try nearby integers
                    found = False
                    for dx_off in [-1, 0, 1]:
                        for dy_off in [-1, 0, 1]:
                            cx = new_x + dx_off
                            cy = new_y + dy_off
                            if dy * cx - dx * cy == c:
                                new_x = cx
                                new_y = cy
                                found = True
                                break
                        if found: break
                    if not found:
                        continue
                
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                
                if any(new_x == p[0] and new_y == p[1] for j, p in enumerate(current_points) if j != idx):
                    continue
                
                new_points = [p[:] for p in current_points]
                new_points[idx] = [new_x, new_y]
                
                if not is_valid(new_points):
                    continue
                    
                new_score = count_ordinary_lines(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                else:
                    delta = new_score - current_score
                    if delta == 0:
                        current_points = new_points
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_points = new_points
                            current_score = new_score
                            improved = True
            else:
                # Strategy 2: Random Move
                idx = random.randint(0, 11)
                old_x, old_y = current_points[idx]
                new_x = old_x + random.randint(-10, 10)
                new_y = old_y + random.randint(-10, 10)
                
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                if any(new_x == p[0] and new_y == p[1] for j, p in enumerate(current_points) if j != idx):
                    continue
                
                new_points = [p[:] for p in current_points]
                new_points[idx] = [new_x, new_y]
                
                if not is_valid(new_points):
                    continue
                    
                new_score = count_ordinary_lines(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                else:
                    delta = new_score - current_score
                    if delta == 0:
                        current_points = new_points
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_points = new_points
                            current_score = new_score
                            improved = True
            
            T *= alpha
            if not improved and T < 0.1:
                break # Stagnation
            
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            break
            
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END
