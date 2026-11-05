import math
import random
import time
from collections import defaultdict

def count_four_lines(points):
    n = len(points)
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normal vector (A, B) = (dy, -dx)
            A = dy
            B = -dx
            C = -(A*x1 + B*y1)
            
            # Normalize
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Canonical form
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            line_id = (A, B, C)
            line_counts[line_id] += 1
            
    four_count = 0
    five_count = 0
    for cnt in line_counts.values():
        if cnt >= 5:
            five_count += 1
        elif cnt == 4:
            four_count += 1
            
    return four_count - 100 * five_count

def construct():
    start_time = time.time()
    time_limit = 115
    
    random.seed(42)
    
    best_points = None
    best_score = -1000
    
    def random_config():
        return [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Start with random
    current_points = random_config()
    current_score = count_four_lines(current_points)
    
    if current_score > best_score:
        best_score = current_score
        best_points = [p[:] for p in current_points]
        
    # Also try starting with 4x4 grid
    grid_pts = [[x, y] for x in range(4) for y in range(4)]
    grid_score = count_four_lines(grid_pts)
    if grid_score > best_score:
        best_score = grid_score
        best_points = [p[:] for p in grid_pts]
        current_points = [p[:] for p in grid_pts]
        current_score = grid_score
        
    # Run SA
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        i = random.randint(0, 15)
        old_x, old_y = current_points[i]
        
        # Move
        if random.random() < 0.5:
            step = random.randint(1, 50)
            new_x = old_x + random.randint(-step, step)
            new_y = old_y + random.randint(-step, step)
        else:
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
        if new_x == old_x and new_y == old_y:
            continue
            
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        is_distinct = True
        for j in range(16):
            if j == i:
                continue
            if current_points[j][0] == new_x and current_points[j][1] == new_y:
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        current_points[i] = [new_x, new_y]
        new_score = count_four_lines(current_points)
        
        delta = new_score - current_score
        if delta > 0:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            if random.random() < math.exp(delta / T):
                current_score = new_score
            else:
                current_points[i] = [old_x, old_y]
                
        T *= cooling_rate
        
    # Final verification
    if best_points:
        # Check validity (no 5 collinear)
        line_counts = defaultdict(int)
        for i in range(16):
            x1, y1 = best_points[i]
            for j in range(i+1, 16):
                x2, y2 = best_points[j]
                dx = x2 - x1
                dy = y2 - y1
                A = dy
                B = -dx
                C = -(A*x1 + B*y1)
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                line_counts[(A, B, C)] += 1
                
        valid = True
        for cnt in line_counts.values():
            if cnt >= 5:
                valid = False
                break
                
        if valid:
            return best_points
    
    # Fallback
    return [[x, y] for x in range(4) for y in range(4)]
