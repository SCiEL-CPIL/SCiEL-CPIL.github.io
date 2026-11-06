import time
import random
import math
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Normalize
    g = math.gcd(abs(a), abs(b))
    if g == 0:
        g = abs(c)
    if g != 0:
        a //= g
        b //= g
        c //= g
    else:
        return None # Same point, shouldn't happen with distinct points
        
    # Canonical sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def count_lines(points):
    # points is a list of [x, y]
    n = len(points)
    if n < 4:
        return 0, True
        
    lines = {}
    valid = True
    
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    score = 0
    for key, pts in lines.items():
        cnt = len(pts)
        if cnt >= 5:
            valid = False
            break
        if cnt == 4:
            score += 1
            
    return score, valid

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    # Initial configuration: 4x4 grid
    # Center it to allow moves in all directions
    init_points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
    
    best_score = -1
    best_points = None
    
    # Evaluate initial
    s, v = count_lines(init_points)
    if v:
        best_score = s
        best_points = [p[:] for p in init_points]
        
    # Local Search
    current_points = [p[:] for p in init_points]
    current_score = best_score
    
    # Define move space
    # Try moving a point to a nearby integer
    # Radius 1-5
    
    while time.time() < deadline:
        improved = False
        # Try random moves
        for _ in range(100): # Iterations per phase
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Generate a new random position nearby
            # Sometimes jump far
            if random.random() < 0.1:
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
            else:
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Ensure distinct
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
                
            # Try move
            current_points[idx] = [new_x, new_y]
            
            s, v = count_lines(current_points)
            
            if v and s > current_score:
                current_score = s
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
                
        if not improved:
            # Restart from best or random
            if best_points:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            else:
                current_points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
                # Ensure distinct
                while len(set(tuple(p) for p in current_points)) < 16:
                    current_points[random.randint(0,15)] = [random.randint(-10, 10), random.randint(-10, 10)]
                s, v = count_lines(current_points)
                if v:
                    current_score = s
                else:
                    current_score = 0
            
            # If stuck for too long, break? No, just keep trying restarts.
            
    if best_points is None:
        return [[0,0]] * 16 # Fallback, though invalid
        
    return best_points
