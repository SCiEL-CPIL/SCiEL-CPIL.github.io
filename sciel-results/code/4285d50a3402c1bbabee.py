import random
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _count_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    Also checks if any line contains 5 or more points (returns -1 if so).
    """
    n = len(points)
    if n < 4:
        return 0
    
    # Dictionary to count points on each line
    # Key: (dx, dy, x0, y0) normalized
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            elif dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                dx = 1
                dy = 1

        # O(N^3) approach is simpler and fast enough for N=16
    four_point_lines = 0
    five_point_lines = 0
    
    # To avoid counting the same line multiple times, we can just iterate all lines defined by pairs
    # and count points on them.
    # But we need to count *distinct* lines.
    
    seen_lines = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            if dx == 0 and dy == 0:
                continue # Distinct points, so not possible
                
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = abs(math.gcd(dx, dy))
                dx //= g
                dy //= g
            
            # Define line key: (dx, dy, x1 - dx * (y1 // dy) if dy != 0 else 0, ...)
            # A robust key for a line in Z^2:
            # A line is defined by Ax + By + C = 0.
            # For direction (dx, dy), normal is (dy, -dx).
            # A = dy, B = -dx.
            # C = -A*x1 - B*y1 = -dy*x1 + dx*y1.
            # Normalize A, B, C such that gcd(A,B,C)=1 and A>0 or (A==0 and B>0).
            
            A = dy
            B = -dx
            C = -A * x1 - B * y1
            
            # Normalize sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            # Reduce by gcd
            g = abs(math.gcd(A, math.gcd(B, C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
                
            line_key = (A, B, C)
            
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            # Count points on this line
            count = 0
            for k in range(n):
                xk, yk = points[k]
                if A * xk + B * yk + C == 0:
                    count += 1
            
            if count == 4:
                four_point_lines += 1
            elif count >= 5:
                five_point_lines += 1
                
    if five_point_lines > 0:
        return -1 # Invalid configuration
        
    return four_point_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Initial configuration: 4x4 grid
    # This is a known valid configuration with score 10
    grid44 = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in grid44]
    best_score = _count_lines(best_points)
    
    # Local Search starting from 4x4 grid
    # We will try to improve the score by moving random points.
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # To escape local optima, we will use a simple hill climbing with random restarts
    # if we get stuck, or just run for a fixed number of steps.
    
    # Strategy:
    # 1. Try to move each point to a better location.
    # 2. If no improvement, try a random perturbation.
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        
        # Try moving a random point
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try a few random moves around the current position or in the whole box
        # Since the box is large, random in box is good for restarts.
        # For local search, maybe try small moves?
        # But integer points are discrete.
        
        # Let's try 20 random moves for this point
        best_move_x, best_move_y = old_x, old_y
        best_move_score = current_score
        
        for _ in range(20):
            # Random in a smaller box around current? Or global?
            # Global is safer to find new structures.
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            # Check distinctness
            is_distinct = True
            for i, p in enumerate(current_points):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            current_points[idx] = [new_x, new_y]
            new_score = _count_lines(current_points)
            
            if new_score > best_move_score:
                best_move_score = new_score
                best_move_x, best_move_y = new_x, new_y
            else:
                # Revert for next trial
                current_points[idx] = [old_x, old_y]
        
        # Apply the best move found
        if best_move_score > current_score:
            current_points[idx] = [best_move_x, best_move_y]
            current_score = best_move_score
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a while, do a random restart from best
            if no_improve_count > max_no_improve:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                no_improve_count = 0
                
                # Optionally perturb best_points slightly before restarting?
                # Or just start fresh from best.
                
    return best_points

# EVOLVE-BLOCK-END
