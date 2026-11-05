import random
import math
import time

def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    if dx == 0:
        dy = 1
    elif dy == 0:
        dx = 1
    else:
        # Normalize A, B for Ax + By + C = 0
        # A = dy, B = -dx
        A = dy
        B = -dx
        g2 = gcd(A, B)
        A //= g2
        B //= g2
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
        # C = -(A*x1 + B*y1)
        C = -(A * x1 + B * y1)
        return (A, B, C)
    # If we didn't go through the general Ax+By+C path (which we should always do for consistency)
    # Actually, the above logic for A,B,C is the standard way.
    # Let's just use (dx, dy, C) where C is derived from normalized direction?
    # No, (A, B, C) is robust.
    # Re-implementing cleanly:
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(dx, dy)
    if g == 0: return None
    dx //= g
    dy //= g
    # Direction vector (dx, dy)
    # Line equation: dy*x - dx*y + c = 0
    # c = dx*y1 - dy*x1
    c = dx * y1 - dy * x1
    # Normalize (dx, dy, c)
    g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if g2 > 0:
        dx //= g2
        dy //= g2
        c //= g2
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
        c = -c
    return (dx, dy, c)

def count_ordinary_lines(points):
    line_counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for count in line_counts.values():
        if count == 1: # A line with exactly 2 points corresponds to 1 pair
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial configuration: 4x3 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_score = count_ordinary_lines(points)
    best_points = points[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_point = points[idx]
        
        # Generate a new random point in a reasonable range
        # Range -20 to 20 is usually sufficient for N=12
        new_point = [random.randint(-20, 20), random.randint(-20, 20)]
        
        # Ensure distinctness
        if new_point in points:
            continue
            
        points[idx] = new_point
        new_score = count_ordinary_lines(points)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = points[:]
                if best_score == 6:
                    # Found theoretical minimum
                    break
        else:
            points[idx] = old_point
            
        T *= cooling
        
    return best_points
