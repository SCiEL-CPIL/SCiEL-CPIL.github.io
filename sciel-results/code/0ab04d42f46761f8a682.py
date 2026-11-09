import random
import itertools
import time

def count_exact_4_lines(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns 0 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return 0
    
    # Check for duplicates
    if len(set(map(tuple, points))) != 16:
        return 0

    # Precompute lines
    # A line is defined by two distinct points.
    # We need to count how many points lie on each unique line.
    # To avoid double counting, we can use a canonical representation of the line.
    # Line equation: ax + by + c = 0
    # We normalize (a, b, c) such that gcd(|a|,|b|,|c|)=1 and a>0 or (a==0 and b>0) or (a==0 and b==0 and c>0)
    
    from math import gcd
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # a*x + b*y + c = 0
        # (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
        a = y2 - y1
        b = -(x2 - x1)
        c = x1 * y2 - x2 * y1
        
        # Normalize
        # Find gcd of a, b, c
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g == 0:
            return None # Should not happen for distinct points
        a //= g
        b //= g
        c //= g
        
        # Ensure canonical sign
        if a < 0:
            a = -a
            b = -b
            c = -c
        elif a == 0:
            if b < 0:
                b = -b
                c = -c
            elif b == 0:
                if c < 0:
                    c = -c
        
        return (a, b, c)

    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    count = 0
    for key, pts_on_line in lines.items():
        k = len(pts_on_line)
        if k > 4:
            return 0 # Invalid configuration
        if k == 4:
            count += 1
            
    return count

def construct():
    # Improved Hill Climbing / Local Search
    
    # Initial configuration: 4x4 grid
    init_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_exact_4_lines(init_points)
    best_points = [p[:] for p in init_points]
    
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    random.seed(42)
    
    # Local search parameters
    neighborhood_radius = 10
    
    while time.time() - start_time < max_time:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = best_points[idx]
        
        # Generate a set of candidate moves
        # 1. Local moves
        # 2. Occasional global jumps (kicks) to escape local optima
        
        # Try a few local moves
        improved = False
        for _ in range(10): # Try 10 random local moves
            if time.time() - start_time > max_time:
                break
                
            new_x = old_x + random.randint(-neighborhood_radius, neighborhood_radius)
            new_y = old_y + random.randint(-neighborhood_radius, neighborhood_radius)
            
            # Keep within bounds
            if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
                continue
                
            best_points[idx] = [new_x, new_y]
            new_score = count_exact_4_lines(best_points)
            
            if new_score > best_score:
                best_score = new_score
                old_x, old_y = new_x, new_y
                improved = True
            else:
                # Revert if not improved
                best_points[idx] = [old_x, old_y]
                
        if not improved:
            # Perform a kick: move a random point to a random location
            # This helps escape local optima
            kick_idx = random.randint(0, 15)
            old_kick_x, old_kick_y = best_points[kick_idx]
            
            new_kick_x = random.randint(-100, 100)
            new_kick_y = random.randint(-100, 100)
            
            best_points[kick_idx] = [new_kick_x, new_kick_y]
            new_score = count_exact_4_lines(best_points)
            
            if new_score > best_score:
                best_score = new_score
            else:
                # Revert kick
                best_points[kick_idx] = [old_kick_x, old_kick_y]
                
    # Final check
    final_score = count_exact_4_lines(best_points)
    if final_score == 0:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
