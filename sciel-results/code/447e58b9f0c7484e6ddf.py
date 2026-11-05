import time
import random
from math import gcd
from itertools import combinations

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the given points.
    Points are a list of [x, y] tuples or lists.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line key to count of points on that line
    # Line key: (dx, dy, c) where dx, dy are normalized direction vector
    # and c = dy*x - dx*y
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            if dx == 0 and dy == 0:
                continue # Should not happen for distinct points
            
            # Ensure unique representation
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
            # Calculate constant c = dy*x1 - dx*y1
            c = dy * x1 - dx * y1
            
            key = (dx, dy, c)
            
            # We need to count how many points are on this line.
            # Incrementing for each pair is tricky because a line with k points 
            # has k*(k-1)/2 pairs. We can't just count pairs.
            # Instead, we can store the set of points or just count pairs and 
            # deduce? No, we need to know if k==2.
            # Better approach: Store the set of indices for each line.
            # But that uses more memory.
            # Alternative: Count pairs per line key.
            # If a line has k points, it contributes k*(k-1)/2 to the pair count.
            # We want to count lines where k=2.
            # If pair_count[key] == 1, then k=2 (since 2*1/2 = 1).
            # If pair_count[key] == 3, then k=3 (since 3*2/2 = 3).
            # If pair_count[key] == 6, then k=4.
            # So we can just count pairs, and check if count == 1.
            
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
    
    ordinary_count = 0
    for key, count in line_counts.items():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Initial candidate: 3x4 grid
    # This is a known starting point.
    # Let's try to find a configuration with 6 ordinary lines.
    
    # Strategy: Local Search
    # Start with a random configuration or a structured one.
    
    def random_points_in_range(r):
        return [[random.randint(-r, r), random.randint(-r, r)] for _ in range(12)]
    
    def valid_points(points):
        if len(points) != 12:
            return False
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
        return True
    
    best_points = None
    best_count = float('inf')
    
    # Try a few structured starts
    # 1. 3x4 Grid
    grid = [[x, y] for x in range(4) for y in range(3)]
    cnt = count_ordinary_lines(grid)
    if cnt < best_count:
        best_count = cnt
        best_points = [p[:] for p in grid]
        
    # 2. 4x3 Grid
    grid2 = [[x, y] for x in range(3) for y in range(4)]
    cnt = count_ordinary_lines(grid2)
    if cnt < best_count:
        best_count = cnt
        best_points = [p[:] for p in grid2]
        
    # 3. 2x6 Grid
    grid3 = [[x, y] for x in range(6) for y in range(2)]
    cnt = count_ordinary_lines(grid3)
    if cnt < best_count:
        best_count = cnt
        best_points = [p[:] for p in grid3]
        
    # Local Improvement
    def improve(points, max_iter=5000):
        pts = [p[:] for p in points]
        current_count = count_ordinary_lines(pts)
        
        # If we already reached the theoretical minimum, we can stop
        if current_count <= 6:
            return pts, current_count
            
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random point to move
            i = random.randint(0, 11)
            old_x, old_y = pts[i]
            
            # Try moving to a nearby integer coordinate
            # Use a small neighborhood to avoid getting stuck in local minima
            # but large enough to escape.
            move_range = 5
            new_x = old_x + random.randint(-move_range, move_range)
            new_y = old_y + random.randint(-move_range, move_range)
            
            # Ensure distinctness
            new_point = (new_x, new_y)
            if new_point in [tuple(p) for p in pts if p is not pts[i]]:
                continue
            
            pts[i] = [new_x, new_y]
            new_count = count_ordinary_lines(pts)
            
            if new_count < current_count:
                current_count = new_count
                improved = True
            else:
                pts[i] = [old_x, old_y]
                
        return pts, current_count

    # Run multiple restarts
    # Start with the best found so far
    if best_points:
        improved_pts, improved_cnt = improve(best_points)
        if improved_cnt < best_count:
            best_count = improved_cnt
            best_points = improved_pts
            
    # Random restarts with increasing range
    random.seed(42)
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
            
        # Start with random points in a small range
        # As restarts increase, we can try larger ranges or perturb best
        if restart % 5 == 0 and best_points:
            # Perturb best
            pts = [p[:] for p in best_points]
            for i in range(12):
                if random.random() < 0.3:
                    pts[i][0] += random.randint(-10, 10)
                    pts[i][1] += random.randint(-10, 10)
        else:
            # Random start
            r = 10 + restart // 5
            pts = random_points_in_range(r)
            
        if not valid_points(pts):
            continue
            
        cnt = count_ordinary_lines(pts)
        if cnt < best_count:
            best_count = cnt
            best_points = [p[:] for p in pts]
            
        # Improve
        if cnt <= best_count + 2: # Only improve if promising
            improved_pts, improved_cnt = improve(pts)
            if improved_cnt < best_count:
                best_count = improved_cnt
                best_points = improved_pts
                
        if best_count <= 6:
            # We found the theoretical minimum
            break

    if best_points is None:
        # Fallback
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
