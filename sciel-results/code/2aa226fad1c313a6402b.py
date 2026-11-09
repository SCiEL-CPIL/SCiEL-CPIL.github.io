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
    # Hardcoded known optimal or near-optimal configuration
    # 3x3 grid + 3 points on a line
    # This specific configuration is known to have few ordinary lines
    # We will use a fast local search to refine it
    
    # Start with 3x3 grid
    base = [[x, y] for x in range(3) for y in range(3)]
    
    # Add 3 points on a line, e.g., x=10
    extra = [[10, 0], [10, 1], [10, 2]]
    
    points = base + extra
    
    # Local Search
    best_score = count_ordinary_lines(points)
    best_points = [p[:] for p in points]
    
    start_time = time.time()
    time_limit = 110
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    while T > T_min and time.time() - start_time < time_limit:
        improved = False
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Try to move point to a line defined by 2 other points
        if random.random() < 0.7:
            others = [i for i in range(12) if i != idx]
            a_idx, b_idx = random.sample(others, 2)
            ax, ay = points[a_idx]
            bx, by = points[b_idx]
            
            dx = bx - ax
            dy = by - ay
            
            if dx == 0 and dy == 0:
                continue
                
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            # Line equation: dy_n * x - dx_n * y = c
            c = dy_n * ax - dx_n * ay
            
            # Find integer point on this line close to old position
            # Parametric: x = ax + k * dx_n, y = ay + k * dy_n
            # We want to find k such that (x,y) is close to (old_x, old_y)
            # k approx = dot(AP, AB) / |AB|^2
            dot_ap_ab = (old_x - ax) * dx_n + (old_y - ay) * dy_n
            dot_ab_ab = dx_n * dx_n + dy_n * dy_n
            
            if dot_ab_ab == 0:
                continue
                
            k_float = dot_ap_ab / dot_ab_ab
            k = round(k_float)
            
            new_x = ax + k * dx_n
            new_y = ay + k * dy_n
            
            # Check validity
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            if any(new_x == p[0] and new_y == p[1] for j, p in enumerate(points) if j != idx):
                continue
                
            new_points = [p[:] for p in points]
            new_points[idx] = [new_x, new_y]
            
            if not is_valid(new_points):
                continue
                
            new_score = count_ordinary_lines(new_points)
            
            if new_score < best_score:
                best_score = new_score
                best_points = new_points
                points = new_points
                improved = True
            else:
                delta = new_score - best_score
                if delta == 0:
                    points = new_points
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        points = new_points
                        improved = True
        else:
            # Random move
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            if any(new_x == p[0] and new_y == p[1] for j, p in enumerate(points) if j != idx):
                continue
                
            new_points = [p[:] for p in points]
            new_points[idx] = [new_x, new_y]
            
            if not is_valid(new_points):
                continue
                
            new_score = count_ordinary_lines(new_points)
            
            if new_score < best_score:
                best_score = new_score
                best_points = new_points
                points = new_points
                improved = True
            else:
                delta = new_score - best_score
                if delta == 0:
                    points = new_points
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        points = new_points
                        improved = True
                        
        T *= alpha
        if not improved and T < 0.1:
            break
            
    return best_points
# EVOLVE-BLOCK-END
