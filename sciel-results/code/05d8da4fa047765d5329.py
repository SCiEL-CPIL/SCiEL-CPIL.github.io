import random
from itertools import combinations
from math import gcd

def count_ordinary_lines(points):
    """Count lines containing exactly 2 points."""
    n = len(points)
    line_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0:
                key = ('v', x1)
            elif dy == 0:
                key = ('h', y1)
            else:
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                a = dy_n
                b = -dx_n
                c = a * x1 + b * y1
                g2 = gcd(abs(a), abs(b))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                key = ('g', a, b, c)
            
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    """Find 12 integer points minimizing ordinary lines via deterministic local search."""
    
    # Start with a 3x4 grid
    best_points = [[x, y] for x in range(3) for y in range(4)]
    best_score = count_ordinary_lines(best_points)
    
    # Deterministic local search
    # Try moving each point to all nearby integer coordinates
    # and accept if it improves the score
    improved = True
    while improved:
        improved = False
        for i in range(12):
            old_x, old_y = best_points[i]
            best_local_score = best_score
            best_local_point = [old_x, old_y]
            
            # Try moves in a small radius
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    # Check bounds (not strictly necessary for score, but good for validity)
                    # Task says |x|, |y| <= 10^9, so small moves are safe.
                    
                    # Check for duplicates
                    dup = False
                    for j in range(12):
                        if j != i and best_points[j][0] == new_x and best_points[j][1] == new_y:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    # Calculate new score
                    temp_points = [p[:] for p in best_points]
                    temp_points[i] = [new_x, new_y]
                    score = count_ordinary_lines(temp_points)
                    
                    if score < best_local_score:
                        best_local_score = score
                        best_local_point = [new_x, new_y]
            
            if best_local_score < best_score:
                best_points[i] = best_local_point
                best_score = best_local_score
                improved = True
    
    return best_points
