import time
import random
import math
from itertools import combinations

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normalize: make dx positive, or if dx==0 make dy positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _score(points):
    """
    Returns (is_valid, count_4_lines).
    is_valid is False if any line has >= 5 points.
    count_4_lines is the number of lines with exactly 4 points.
    """
    n = len(points)
    if n != 16:
        return False, 0
    
    line_counts = {}
    # 16 points -> 120 pairs
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
            
    count_4 = 0
    for cnt in line_counts.values():
        if cnt >= 5:
            return False, 0
        if cnt == 4:
            count_4 += 1
            
    return True, count_4

def _local_search(points, deadline):
    """Improve points via local search with random restarts/jumps."""
    best = [list(p) for p in points]
    valid, best_score = _score(best)
    if not valid:
        best_score = -1 # Invalid
    
    # If initial is invalid, try to fix it immediately
    if not valid:
         # Just return best valid found later, or this if it becomes valid
         pass

    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point slightly
        # We do a full pass over indices if time permits, or just random steps
        # Let's do random steps for speed and exploration
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        x, y = best[idx]
        
        # Random perturbation
        # Mix of small and large steps
        if random.random() < 0.5:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
        else:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            
        nx = x + dx
        ny = y + dy
        
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
            
        new_points = [list(p) for p in best]
        new_points[idx] = [nx, ny]
        
        # Check distinctness
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
            
        is_valid, score = _score(new_points)
        
        if is_valid:
            if score > best_score:
                best = new_points
                best_score = score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Random restart or large jump
            # Let's just continue, the random nature will eventually jump out
            # Or we can reset to a known good config? 
            # Better to just keep going with larger steps
            pass
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    # Initial configuration: 4x4 grid
    # This is known to have 10 lines of 4 points and is valid.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [list(p) for p in initial_points]
    best_score = -1
    
    valid, score = _score(best_points)
    if valid:
        best_score = score
    
    # Run local search
    # We will run multiple phases or just one long one?
    # One long one with random restarts is fine.
    
    current_points = [list(p) for p in initial_points]
    
    # To be safe, let's try a few different starting points if time permits
    # But 4x4 is a strong local optimum.
    # Let's just run the search from 4x4.
    
    result_points, result_score = _local_search(current_points, deadline)
    
    if result_score > best_score:
        best_points = result_points
        best_score = result_score
        
    # Check if we have time for another strategy?
    # Maybe a random start?
    if time.time() < deadline - 10:
        random_points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
        # Ensure distinct
        while len(set(map(tuple, random_points))) < 16:
             random_points[random.randint(0,15)] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
             
        r_points, r_score = _local_search(random_points, deadline)
        if r_score > best_score:
            best_points = r_points
            best_score = r_score

    # Final validation check
    is_valid, final_score = _score(best_points)
    if not is_valid:
        # Fallback to 4x4 grid which is always valid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
