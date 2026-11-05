import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    # Initial configuration: 4x4 grid
    # This gives 10 four-point lines (4 horiz, 4 vert, 2 diag)
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def gcd2(x, y):
        while y:
            x, y = y, x % y
        return x

    def score_fast(points):
        n = 16
        pts = points
        processed = set()
        count4 = 0
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                a = y2 - y1
                b = x1 - x2
                c = x2 * y1 - x1 * y2
                
                # Normalize
                g = gcd2(abs(a), abs(b))
                if g == 0:
                    g = abs(c)
                else:
                    g = gcd2(g, abs(c))
                
                if g == 0:
                    continue
                a //= g
                b //= g
                c //= g
                
                if a < 0 or (a == 0 and b < 0):
                    a, b, c = -a, -b, -c
                key = (a, b, c)
                
                if key in processed:
                    continue
                processed.add(key)
                
                cnt = 0
                for k in range(n):
                    xk, yk = pts[k]
                    if a * xk + b * yk + c == 0:
                        cnt += 1
                        if cnt >= 5:
                            return -1
                if cnt == 4:
                    count4 += 1
        return count4

    best_points = [p[:] for p in points]
    best_score = score_fast(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simulated Annealing Parameters
    T_start = 2.0
    T_end = 0.01
    total_time = time_limit
    max_offset = 20  # Larger jumps to explore more of the space
    
    iterations = 0
    
    while time.time() - start_time < time_limit:
        # Calculate current temperature
        elapsed = time.time() - start_time
        progress = elapsed / total_time
        T = T_start * (1 - progress) + T_end * progress
        
        # Random move
        idx = random.randint(0, 15)
        old_p = current_points[idx]
        
        # Try a few random offsets
        for _ in range(5): # Reduced inner loop to allow more outer iterations
            dx = random.randint(-max_offset, max_offset)
            dy = random.randint(-max_offset, max_offset)
            if dx == 0 and dy == 0:
                continue
            new_p = [old_p[0] + dx, old_p[1] + dy]
            if not (-1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000):
                continue
            
            # Check distinctness
            is_distinct = True
            for k, p in enumerate(current_points):
                if k == idx:
                    continue
                if p[0] == new_p[0] and p[1] == new_p[1]:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            
            # Apply move
            current_points[idx] = new_p
            
            # Score
            s = score_fast(current_points)
            
            if s > current_score:
                # Always accept better
                current_score = s
                if s > best_score:
                    best_score = s
                    best_points = [p[:] for p in current_points]
            else:
                # Accept worse with probability exp((new - old) / T)
                if s != -1: # Only accept if valid
                    delta = s - current_score
                    if random.random() < math.exp(delta / T):
                        current_score = s
                    else:
                        # Revert
                        current_points[idx] = old_p
                else:
                    # Revert invalid
                    current_points[idx] = old_p
        
        iterations += 1

    return best_points
