import time
import random
import math

def _score(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Returns 0 if any line has 5 or more points (invalid).
    Returns -1 if points are not distinct.
    """
    n = len(points)
    # Check distinctness
    if len(set(map(tuple, points))) != n:
        return -1
        
    # Map from line key to set of points on that line
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line equation: ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize the line equation
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue # Should not happen for distinct points
                
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    four_lines = 0
    for pts in line_points.values():
        k = len(pts)
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0 # Invalid configuration: 5 or more collinear
            
    return four_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Initial configuration: 4x4 grid scaled by 3 to avoid immediate boundary issues and allow jitter
    # Points: (0,0) to (9,9) step 3
    # This has 10 lines.
    scale = 3
    points = [[i * scale, j * scale] for i in range(4) for j in range(4)]
    
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    n = 16
    max_coord = 1000
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Counters for restarts
    stall_counter = 0
    max_stall = 20000
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        
        # Generate a new position
        # Mixed step sizes to escape local minima
        r = random.random()
        if r < 0.5:
            # Small step
            step = random.choice([1, -1, 2, -2, 3, -3])
        elif r < 0.8:
            # Medium step
            step = random.randint(-10, 10)
        else:
            # Large jump
            step = random.randint(-50, 50)
            
        dx = step
        dy = random.randint(-abs(step), abs(step)) if step != 0 else random.choice([1, -1])
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if not (-max_coord <= new_x <= max_coord and -max_coord <= new_y <= max_coord):
            continue
            
        new_point = [new_x, new_y]
        
        # Check distinctness
        is_distinct = True
        for k in range(n):
            if k != idx and points[k][0] == new_x and points[k][1] == new_y:
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        # Apply move
        points[idx] = new_point
        new_score = _score(points)
        
        if new_score > current_score:
            # Accept better move
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                stall_counter = 0
            else:
                stall_counter += 1
        else:
            # Accept worse move with probability exp(-delta/T)
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # If score is invalid (-1 or 0 when it was >0), reject strongly
                if new_score < 0:
                    accept = False
                elif new_score == 0 and current_score > 0:
                    # Strong penalty for breaking valid config into invalid (5-collinear)
                    # exp(-large/T)
                    accept = random.random() < math.exp(-10.0 / T)
                else:
                    accept = random.random() < math.exp(-delta / T)
                    
            if accept:
                stall_counter += 1
            else:
                # Revert
                points[idx] = [old_x, old_y]
                stall_counter += 1
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Restart logic: If stuck for a while, reheat and possibly reset to best
        if stall_counter > max_stall:
            T = 5.0
            stall_counter = 0
            # 50% chance to reset to best configuration to explore from there again
            if random.random() < 0.5:
                points = [p[:] for p in best_points]
                current_score = best_score
            else:
                # Randomize one point significantly
                idx2 = random.randint(0, n - 1)
                points[idx2] = [random.randint(-100, 100), random.randint(-100, 100)]
                # Ensure distinct
                while not _distinct(points):
                     points[idx2] = [random.randint(-100, 100), random.randint(-100, 100)]
                current_score = _score(points)
                if current_score < 0:
                    points = [p[:] for p in best_points]
                    current_score = best_score

    return best_points

def _distinct(points):
    return len(set(map(tuple, points))) == len(points)
