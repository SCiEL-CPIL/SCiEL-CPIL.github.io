import random
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normalize direction
            g = __import__('math').gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Ensure unique representation
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            
            # Line equation: ax + by = c
            # Normal vector (dy, -dx)
            a = dy
            b = -dx
            c = a * p1[0] + b * p1[1]
            key = (a, b, c)
            if key not in lines:
                lines[key] = [i, j]
            else:
                lines[key].append(i)
                lines[key].append(j)
                # Remove duplicates
                lines[key] = list(set(lines[key]))
    
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initial 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_score = count_ordinary_lines(points)
    best_points = points[:]
    best_score = current_score
    
    # Local search
    while time.time() < deadline:
        # Pick a random point
        idx = random.randint(0, 11)
        old_p = best_points[idx][:]
        
        # Try to move to a nearby point
        # Random step in x and y
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_p = [old_p[0] + dx, old_p[1] + dy]
        
        # Check bounds
        if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
            continue
            
        # Check distinctness
        new_points = best_points[:]
        new_points[idx] = new_p
        if len(set(map(tuple, new_points))) < 12:
            continue
            
        # Check not all collinear
        # (Implicitly handled by score, but good to be sure)
        
        score = count_ordinary_lines(new_points)
        
        if score < best_score:
            best_score = score
            best_points = new_points
            
        # If we reached the theoretical minimum, stop
        if best_score == 6:
            break
            
    return best_points
