import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def evaluate(pts):
        line_map = defaultdict(set)
        n = len(pts)
        
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                A = y2 - y1
                B = -(x2 - x1)
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0:
                    continue # Should not happen for distinct points
                
                # Normalize
                g = math.gcd(abs(A), abs(B))
                if g == 0:
                    continue
                A //= g
                B //= g
                # C normalization: C must be divisible by g?
                # A*x + B*y + C = 0. If we divide A,B by g, C should be divided by g only if C is divisible by g?
                # Actually, the line equation is not unique.
                # Better: Use the pair (slope, intercept) or (normal vector, constant).
                # Let's stick to A,B,C but normalize properly.
                # The primitive integer normal vector is (A/g, B/g).
                # The constant term C/g might not be integer if C is not divisible by g.
                # Example: (0,0) and (1,1). A=1, B=-1, C=0. g=1. Key (1,-1,0).
                # Example: (0,0) and (2,2). A=2, B=-2, C=0. g=2. A=1, B=-1, C=0.
                # So we must divide C by g as well?
                # If A=2, B=2, C=1. g=2. A=1, B=1, C=0.5. Not integer.
                # So we cannot just divide C by g.
                # We need the unique representation of the line.
                # The line is determined by the direction vector (dx, dy) and one point.
                # Or by the normalized normal vector (A, B) and the value k = A*x + B*y.
                # Let (A, B) be the primitive normal vector.
                # Then k = A*x1 + B*y1.
                # Key = (A, B, k).
                
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                dx //= g
                dy //= g
                
                # Normal vector (A, B) perpendicular to (dx, dy)
                A = dy
                B = -dx
                
                # Normalize (A, B) to a canonical form
                # Ensure A > 0 or (A == 0 and B > 0)
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                
                k = A * x1 + B * y1
                
                key = (A, B, k)
                line_map[key].add(i)
                line_map[key].add(j)
        
        count_4 = 0
        for key, indices in line_map.items():
            if len(indices) >= 5:
                return -1, 0
            if len(indices) == 4:
                count_4 += 1
        
        return 1, count_4

    _, best_score = evaluate(points)
    best_points = [p[:] for p in points]
    current_score = best_score
    current_points = [p[:] for p in points]
    
    T = 20.0
    T_min = 0.5
    cooling_rate = 0.9995
    
    step = 0
    while time.time() - start_time < time_limit:
        new_points = [p[:] for p in current_points]
        idx = random.randint(0, 15)
        
        # Choose move size
        # 80% small, 15% medium, 5% large
        r = random.random()
        if r < 0.8:
            max_delta = 2
        elif r < 0.95:
            max_delta = 5
        else:
            max_delta = 20
            
        dx = random.randint(-max_delta, max_delta)
        dy = random.randint(-max_delta, max_delta)
        
        if dx == 0 and dy == 0:
            continue
            
        nx = new_points[idx][0] + dx
        ny = new_points[idx][1] + dy
        
        if not (-1000 <= nx <= 1000 and -1000 <= ny <= 1000):
            continue
            
        new_point = [nx, ny]
        is_distinct = True
        for i, p in enumerate(new_points):
            if i != idx and p == new_point:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        new_points[idx] = new_point
        
        valid, score = evaluate(new_points)
        
        if valid:
            delta = score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points = new_points
                current_score = score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        step += 1
        # Restart to best every 1000 steps to escape local optima
        if step % 1000 == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = 10.0
            
    return best_points
