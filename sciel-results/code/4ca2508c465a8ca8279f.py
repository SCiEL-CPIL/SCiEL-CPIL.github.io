import time
import random
import math

def count_lines(points):
    """
    Returns (score, is_valid)
    score: number of lines with exactly 4 points
    is_valid: True if no line has 5 or more points
    """
    n = len(points)
    # Use a set of canonical line representations to avoid double counting
    # A line can be represented by (A, B, C) from Ax + By + C = 0
    # Normalized: GCD(A,B,C)=1, and A>0 or (A==0 and B>0)
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            # Vector (dx, dy)
            dx = x2 - x1
            dy = y2 - y1
            
            # Line coefficients: dy*x - dx*y + C = 0
            # C = dx*y1 - dy*x1
            A = dy
            B = -dx
            C = dx*y1 - dy*x1
            
            # Normalize (A, B, C)
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                # Points are identical, which shouldn't happen if distinct
                continue
            A //= g
            B //= g
            C //= g
            
            # Ensure canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = 0
            
            # Count points on this line
            # We already know i and j are on it.
            # Check others
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if A*x3 + B*y3 + C == 0:
                    count += 1
            
            lines[key] = count
            
    valid = True
    score = 0
    for count in lines.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            score += 1
            
    return score, valid

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    best_score = -1
    best_points = None
    
    # Initial seed: 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    score, valid = count_lines(initial_points)
    if valid:
        best_score = score
        best_points = [p[:] for p in initial_points]
    
    # Local Search
    points = [p[:] for p in initial_points]
    current_score = best_score
    
    # To diversify, we can try random restarts if stuck
    stuck_counter = 0
    restart_count = 0
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        orig_x, orig_y = points[idx]
        
        # Random move: shift by small amount
        # We want to stay within bounds [-1000, 1000]
        # But we can also try larger jumps occasionally
        if random.random() < 0.1:
            # Large jump
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        else:
            # Small jump
            new_x = orig_x + random.randint(-3, 3)
            new_y = orig_y + random.randint(-3, 3)
            
        # Ensure bounds
        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
            continue
            
        # Ensure distinctness
        is_distinct = True
        for k in range(16):
            if k != idx:
                if points[k][0] == new_x and points[k][1] == new_y:
                    is_distinct = False
                    break
        if not is_distinct:
            continue
            
        # Apply move
        points[idx] = [new_x, new_y]
        
        new_score, new_valid = count_lines(points)
        
        if new_valid:
            if new_score > current_score:
                current_score = new_score
                stuck_counter = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
            else:
                # Revert if not improved (strict hill climbing)
                # Or accept with some probability for simulated annealing?
                # Let's try strict first, then add noise if stuck.
                points[idx] = [orig_x, orig_y]
                stuck_counter += 1
                
                if stuck_counter > 100:
                    # Try a random restart from best or random
                    if random.random() < 0.5:
                        # Restart from best
                        points = [p[:] for p in best_points]
                        current_score = best_score
                    else:
                        # Random restart
                        points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
                        # Check validity of random restart
                        s, v = count_lines(points)
                        if v:
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_points = [p[:] for p in points]
                        else:
                            # If invalid, revert to best
                            points = [p[:] for p in best_points]
                            current_score = best_score
                    stuck_counter = 0
                    restart_count += 1
        else:
            # Invalid, revert
            points[idx] = [orig_x, orig_y]
            stuck_counter += 1
            if stuck_counter > 100:
                 # Same restart logic
                 if random.random() < 0.5:
                     points = [p[:] for p in best_points]
                     current_score = best_score
                 else:
                     points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
                     s, v = count_lines(points)
                     if v:
                         current_score = s
                         if s > best_score:
                             best_score = s
                             best_points = [p[:] for p in points]
                     else:
                         points = [p[:] for p in best_points]
                         current_score = best_score
                 stuck_counter = 0

    if best_points is None:
        # Fallback to seed
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-END
