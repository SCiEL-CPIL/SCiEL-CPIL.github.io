import time
import random
import math

def _score(points):
    """
    Fast evaluation using pair counting.
    Returns the number of lines with exactly 4 points.
    Returns -1 if invalid (5+ collinear) or points not distinct.
    """
    n = len(points)
    if n != 16:
        return -1
        
    # Check distinctness
    if len(set(map(tuple, points))) != n:
        return -1
        
    # Map from line key to pair count
    # A line with k points will have C(k, 2) pairs.
    # k=4 -> 6 pairs. k=5 -> 10 pairs.
    line_pair_counts = {}
    
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
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
            
    four_lines = 0
    for count in line_pair_counts.values():
        if count == 6: # C(4,2)
            four_lines += 1
        elif count >= 10: # C(5,2) or more -> Invalid
            return -1 # Invalid configuration
            
    return four_lines

def _local_polish(points, deadline):
    """
    Hill climbing local search to improve the current configuration.
    Returns the improved points and score.
    """
    current_points = [p[:] for p in points]
    current_score = _score(current_points)
    n = 16
    max_coord = 1000
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try moving each point slightly
        for idx in range(n):
            old_x, old_y = current_points[idx]
            # Try small steps in 8 directions + stay
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-max_coord <= new_x <= max_coord and -max_coord <= new_y <= max_coord):
                        continue
                    new_point = [new_x, new_y]
                    # Check distinctness
                    is_distinct = True
                    for k in range(n):
                        if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                    current_points[idx] = new_point
                    new_score = _score(current_points)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        # Don't break, keep looking for further improvements from this new state
                    else:
                        # Revert
                        current_points[idx] = [old_x, old_y]
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Initial configuration: 4x4 grid
    # This has score 10.
    # We scale it slightly to give room for movement, but the grid structure is the key.
    # Let's stick to the small integer grid to maximize line density initially.
    points = [[i, j] for i in range(4) for j in range(4)]
    
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    n = 16
    max_coord = 1000
    
    # Simulated Annealing parameters
    T = 15.0
    T_min = 0.05
    cooling_rate = 0.9998
    
    # Counters for restarts
    stall_counter = 0
    max_stall = 5000
    
    # Run local polish immediately on the initial grid to see if we can easily improve
    points, current_score = _local_polish(points, deadline)
    if current_score > best_score:
        best_score = current_score
        best_points = [p[:] for p in points]
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        
        # Generate a new position
        # Mixed step sizes to escape local minima
        r = random.random()
        if r < 0.6:
            # Small step
            step = random.choice([1, -1, 2, -2, 3, -3])
        elif r < 0.85:
            # Medium step
            step = random.randint(-8, 8)
        else:
            # Large jump
            step = random.randint(-30, 30)
            
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
                # Trigger a local polish when we find a new global best
                # This helps exploit the new basin
                if time.time() < deadline - 5.0: # Reserve time
                    points, current_score = _local_polish(points, deadline)
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
            else:
                stall_counter += 1
        else:
            # Accept worse move with probability exp(-delta/T)
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # If score is invalid (-1), reject strongly unless T is high
                if new_score < 0:
                    # Allow invalid states early on to navigate, but penalize heavily
                    accept = random.random() < math.exp(-20.0 / T)
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
            T = 10.0
            stall_counter = 0
            # 50% chance to reset to best configuration to explore from there again
            if random.random() < 0.5:
                points = [p[:] for p in best_points]
                current_score = best_score
            else:
                # Randomize one point significantly to kick out of local minimum
                idx2 = random.randint(0, n - 1)
                points[idx2] = [random.randint(-50, 50), random.randint(-50, 50)]
                # Ensure distinct
                while True:
                    distinct = True
                    for k in range(n):
                        if k != idx2 and points[k] == points[idx2]:
                            distinct = False
                            break
                    if distinct:
                        break
                    points[idx2] = [random.randint(-50, 50), random.randint(-50, 50)]
                current_score = _score(points)
                if current_score < 0:
                    points = [p[:] for p in best_points]
                    current_score = best_score

    return best_points

# Keep helper if needed, though _score now handles distinctness check implicitly by returning -1
# But we don't call _distinct separately anymore in the main loop to save time.

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(pts)
