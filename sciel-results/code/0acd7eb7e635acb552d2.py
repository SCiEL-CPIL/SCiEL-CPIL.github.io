import math
import random
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x1 * y2 - x2 * y1
            
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary_count = 0
    for count in line_counts.values():
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 100 # 20s margin
    
    # Initial configuration: 4x3 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simple Local Search
    # Try to move points to reduce ordinary lines
    # We use a small neighborhood to keep it fast
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point to a random nearby location
        # We do a few passes
        for _ in range(10):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 11)
            ox, oy = current_points[idx]
            
            # Try a few random offsets
            for _ in range(5):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                nx = ox + dx
                ny = oy + dy
                
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                
                # Check distinctness
                is_distinct = True
                for i, p in enumerate(current_points):
                    if i == idx:
                        continue
                    if p[0] == nx and p[1] == ny:
                        is_distinct = False
                        break
                
                if not is_distinct:
                    continue
                
                new_points = [p[:] for p in current_points]
                new_points[idx] = [nx, ny]
                
                new_score = count_ordinary_lines(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                        if best_score == 6:
                            return best_points
                        no_improve_count = 0
                    break # Move on to next point/index
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with a new random configuration or perturb heavily
                # Let's try a different grid or random start
                # For now, just break if stuck too long to ensure return
                break
        
    return best_points
