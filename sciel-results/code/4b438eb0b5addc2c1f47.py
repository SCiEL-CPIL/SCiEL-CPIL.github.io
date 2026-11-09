import time
import random
import itertools

def count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Map point to index
    point_to_idx = {tuple(p): i for i, p in enumerate(points)}
    
    # Count lines
    # A line is defined by two points.
    # We need to find lines with exactly 2 points.
    
    # For each pair of points, determine the line.
    # Then for each line, count how many points lie on it.
    
    # To avoid floating point issues, use cross product for collinearity.
    # Or use a dictionary of lines represented by (A, B, C) for Ax+By+C=0.
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x2 = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - y2 * x1
            
            # Normalize the line equation
            # Find gcd of A, B, C
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            # Ensure a canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = 0
            
            # Count points on this line
            # We can just increment a counter for the pair, but we need unique points.
            # Better: store the set of points or just count unique points.
            # Since n is small (12), we can just check all points for each unique line.
            
    # Now count points on each line
    ordinary_count = 0
    for key in lines:
        A, B, C = key
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial configuration: 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_score = count_ordinary(points)
    best_points = points[:]
    best_score = current_score
    
    # Local search
    # Try moving one point to a new random location
    # Or try small shifts
    
    while time.time() < deadline:
        improved = False
        for i in range(len(points)):
            if time.time() > deadline:
                break
            
            orig_p = points[i]
            
            # Try a few random moves
            for _ in range(50):
                # Random new position
                # Keep within bounds
                new_x = random.randint(-10, 10)
                new_y = random.randint(-10, 10)
                
                # Ensure distinct
                if [new_x, new_y] in points:
                    continue
                
                points[i] = [new_x, new_y]
                score = count_ordinary(points)
                
                if score < current_score:
                    current_score = score
                    if score < best_score:
                        best_score = score
                        best_points = points[:]
                    improved = True
                else:
                    points[i] = orig_p
                    
            # Also try small shifts
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_p[0] + dx
                    new_y = orig_p[1] + dy
                    if [new_x, new_y] in points:
                        continue
                    points[i] = [new_x, new_y]
                    score = count_ordinary(points)
                    if score < current_score:
                        current_score = score
                        if score < best_score:
                            best_score = score
                            best_points = points[:]
                        improved = True
                    else:
                        points[i] = orig_p
        
        if not improved:
            # Try a larger random jump
            i = random.randint(0, len(points) - 1)
            orig_p = points[i]
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            if [new_x, new_y] not in points:
                points[i] = [new_x, new_y]
                score = count_ordinary(points)
                if score < current_score:
                    current_score = score
                    if score < best_score:
                        best_score = score
                        best_points = points[:]
                else:
                    points[i] = orig_p
            # If stuck, restart with a different initial config?
            # For now, just continue searching.
            
    return best_points
