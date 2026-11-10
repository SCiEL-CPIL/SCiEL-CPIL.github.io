import time
import random
import math
from itertools import combinations

# Deadline guard: 140s safety margin for 150s limit
DEADLINE = time.time() + 140

def _count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the given points.
    Points are a list of tuples (x, y).
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Map from line key to count of points on that line
    # Line key: (dx, dy, c) where line is dx*x + dy*y = c
    # We need to normalize (dx, dy) such that gcd(dx,dy)=1 and dx>0 or (dx==0 and dy>0)
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue # Duplicate points, shouldn't happen if distinct
                
            # Normalize direction vector
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                dx_n, dy_n = dx, dy
                
            # Canonical direction: dx > 0 or (dx == 0 and dy > 0)
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Calculate c = dx*x + dy*y
            # To keep integers, use the normalized direction
            c = dx_n * x1 + dy_n * y1
            
            key = (dx_n, dy_n, c)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    ordinary = 0
    for count in line_counts.values():
        if count == 1:
            ordinary += 1
            
    return ordinary

def _random_points_in_range(n, limit=100):
    pts = []
    seen = set()
    while len(pts) < n:
        x = random.randint(-limit, limit)
        y = random.randint(-limit, limit)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return pts

def _local_search(initial_points, max_time=10):
    """
    Hill climbing with random restarts and restarts on plateau.
    """
    best_points = initial_points[:]
    best_score = _count_ordinary_lines(best_points)
    
    if best_score == 6:
        return best_points, best_score
        
    current_points = initial_points[:]
    current_score = best_score
    
    start_time = time.time()
    stall_count = 0
    max_stall = 2000 # Restart if no improvement for 2000 steps
    
    while time.time() - start_time < max_time:
        # Pick a random point to move
        idx = random.randint(0, len(current_points) - 1)
        old_x, old_y = current_points[idx]
        
        # Try moving to a random nearby or far location
        # Strategy: Try to snap to a line defined by 2 other points
        # Pick 2 other points
        others = [i for i in range(len(current_points)) if i != idx]
        i1, i2 = random.sample(others, 2)
        x1, y1 = current_points[i1]
        x2, y2 = current_points[i2]
        
        # Candidate point on line through p1, p2
        # Line: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        # A = y2-y1, B = -(x2-x1), C = (x2-x1)y1 - (y2-y1)x1
        A = y2 - y1
        B = -(x2 - x1)
        C = (x2 - x1) * y1 - (y2 - y1) * x1
        
        # Find integer point on this line close to old position or random
        # Parametric: x = x1 + t*dx, y = y1 + t*dy
        dx = x2 - x1
        dy = y2 - y1
        g = math.gcd(abs(dx), abs(dy))
        if g > 0:
            dx_step = dx // g
            dy_step = dy // g
        else:
            dx_step, dy_step = dx, dy
            
        # Try a few integer steps from p1
        improved = False
        for t in range(-10, 11):
            if t == 0: continue # p1
            if t == g: continue # p2 (if g is step size to p2, actually t=1 is p2 if g=1)
            # Actually, p2 is at t = g if we used reduced step? No.
            # p2 = p1 + (dx, dy). If step is (dx/g, dy/g), then p2 is at t=g.
            if t == g: continue
            
            nx = x1 + t * dx_step
            ny = y1 + t * dy_step
            
            # Check bounds
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
                
            # Check distinctness
            new_pt = (nx, ny)
            if new_pt in current_points:
                continue
                
            # Calculate new score
            test_points = current_points[:]
            test_points[idx] = new_pt
            
            # Optimization: Only recalculate if it looks promising? 
            # For n=12, full recalc is fast (~66 pairs).
            new_score = _count_ordinary_lines(test_points)
            
            if new_score < current_score:
                current_points = test_points
                current_score = new_score
                improved = True
                stall_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
                break
        if not improved:
            stall_count += 1
            if stall_count > max_stall:
                # Random restart
                current_points = _random_points_in_range(12, limit=50)
                current_score = _count_ordinary_lines(current_points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
                stall_count = 0
                max_stall = 5000 # Increase stall tolerance after restart? Or keep same.

    return best_points, best_score

def construct():
    # Start with a 3x4 grid, known to be a decent seed
    seed_points = [(x, y) for x in range(4) for y in range(3)]
    
    # Also try a 2x6 grid
    seed_points_2 = [(x, y) for x in range(6) for y in range(2)]
    
    # Also try random
    seed_points_3 = _random_points_in_range(12, limit=20)
    
    best_overall_points = seed_points
    best_overall_score = _count_ordinary_lines(seed_points)
    
    # Run local search on each seed
    for seed in [seed_points, seed_points_2, seed_points_3]:
        if time.time() > DEADLINE - 10:
            break
        pts, score = _local_search(seed, max_time=30)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_points = pts
        if best_overall_score == 6:
            break
            
    # Convert to list of lists
    return [[p[0], p[1]] for p in best_overall_points]
