import random
import time
import itertools

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    ordinary = 0
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            # Line defined by p1, p2
            # Normalize line representation
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Simplify dx, dy by gcd
            from math import gcd
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Ensure consistent sign
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line equation: dy*x - dx*y = c
            c = dy * p1[0] - dx * p1[1]
            line_key = (dx, dy, c)
            
            if line_key not in lines:
                lines.add(line_key)
                # Count points on this line
                count = 0
                for k in range(n):
                    pk = points[k]
                    if dy * pk[0] - dx * pk[1] == c:
                        count += 1
                if count == 2:
                    ordinary += 1
    return ordinary

def construct():
    # Start with a known good configuration
    # 3x3 grid has 12 ordinary lines
    base_points = [[x, y] for x in range(3) for y in range(3)]
    best_score = count_ordinary_lines(base_points)
    best_points = base_points[:]
    
    deadline = time.time() + 140
    
    # Try to improve by adding points or moving points
    # We need exactly 12 points.
    # Let's try to find a 12-point configuration on a small grid.
    # Random search on 12 points in [-5, 5]
    
    while time.time() < deadline:
        # Generate random 12 points
        pts = []
        seen = set()
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        
        # Check if all collinear
        if len(set(pts)) < 12:
            continue
            
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = pts[:]
            
        if best_score == 6: # Theoretical minimum
            break
            
    return best_points
