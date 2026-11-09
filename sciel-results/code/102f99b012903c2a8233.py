# EVOLVE-BLOCK-START
import math
import random
import time
from collections import defaultdict

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    lines = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def is_valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in points):
        return False
    # Check not all collinear
    (x0, y0) = points[0]
    (x1, y1) = points[1]
    (x2, y2) = points[2]
    if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
        all_collinear = True
        for i in range(3, 12):
            (x3, y3) = points[i]
            if (x1 - x0) * (y3 - y0) != (y1 - y0) * (x3 - x0):
                all_collinear = False
                break
        if all_collinear:
            return False
    return True

def construct():
    best_points = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 110
    
    # Strategy: Try multiple random restarts with a fast local search.
    # The key is to use a smarter move set: 
    # 1. Move a point to lie on a line defined by two other points (if integer).
    # 2. Move a point to lie on a line defined by one other point and a "direction" that creates a 3-point line.
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random starting configuration
        # Use a mix of grid and random
        if restart % 2 == 0:
            # Perturbed 3x4 grid
            points = []
            for x in range(4):
                for y in range(3):
                    points.append([x + random.randint(-1, 1), y + random.randint(-1, 1)])
        else:
            # Random points in a box
            points = []
            for _ in range(12):
                points.append([random.randint(-20, 20), random.randint(-20, 20)])
        
        # Ensure distinctness
        for i in range(12):
            if i > 0:
                while points[i] in points[:i]:
                    points[i][0] += 1
        
        if not is_valid(points):
            continue
            
        current_points = [p[:] for p in points]
        current_score = count_ordinary_lines(current_points)
        
        # Local Search
        # We will perform a series of "snap to line" moves.
        # For each point i, and each pair (j, k), check if there is an integer point on line jk
        # that is close to i, and move i there if it improves the score.
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Shuffle order to avoid bias
            indices = list(range(12))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > time_limit:
                    break
                
                # Try moving point i to a position on a line defined by j, k
                # We sample a few pairs (j, k)
                for _ in range(5):
                    j, k = random.sample([x for x in range(12) if x != i], 2)
                    xj, yj = current_points[j]
                    xk, yk = current_points[k]
                    
                    dx = xk - xj
                    dy = yk - yj
                    
                    if dx == 0 and dy == 0:
                        continue
                    
                    # Line equation: dy*x - dx*y = c
                    c = dy * xj - dx * yj
                    
                    # We want to find an integer point (nx, ny) on this line
                    # that is "good". A good point is one that is currently close to point i,
                    # or simply one that results in a lower score.
                    # To be efficient, we only check if the line jk already has >= 2 points.
                    # If it has 2, moving i to it makes it 3.
                    # If it has 1 (impossible as we pick j,k from points), it's always >=2.
                    # Wait, the line is defined by j and k. So it contains at least j and k.
                    
                    # Find closest integer point on line jk to current_points[i]
                    xi, yi = current_points[i]
                    
                    # Projection
                    if dx == 0:
                        # Vertical line x = xj
                        nx = xj
                        ny = yi
                    elif dy == 0:
                        # Horizontal line y = yj
                        nx = xi
                        ny = yj
                    else:
                        # General case
                        # We can parametrize the line.
                        # Or simply check nearby integer points?
                        # No, we need the exact closest.
                        # t = ((xi-xj)*dx + (yi-yj)*dy) / (dx^2 + dy^2)
                        dot = (xi - xj) * dx + (yi - yj) * dy
                        dist_sq = dx*dx + dy*dy
                        t = dot / dist_sq
                        proj_x = xj + t * dx
                        proj_y = yj + t * dy
                        
                        nx = round(proj_x)
                        ny = round(proj_y)
                        
                        # Verify (nx, ny) is on the line
                        if dy * nx - dx * ny != c:
                            # Check neighbors
                            found = False
                            for dx_off in [-1, 0, 1]:
                                for dy_off in [-1, 0, 1]:
                                    cx = nx + dx_off
                                    cy = ny + dy_off
                                    if dy * cx - dx * cy == c:
                                        nx = cx
                                        ny = cy
                                        found = True
                                        break
                                if found: break
                            if not found:
                                continue
                            
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    
                    # Check collision
                    if any(nx == p[0] and ny == p[1] for idx, p in enumerate(current_points) if idx != i):
                        continue
                        
                    # Check if it's the same point
                    if nx == xi and ny == yi:
                        continue
                        
                    # Try the move
                    new_points = [p[:] for p in current_points]
                    new_points[i] = [nx, ny]
                    
                    if not is_valid(new_points):
                        continue
                        
                    new_score = count_ordinary_lines(new_points)
                    
                    if new_score < current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        break # Restart the inner loop for this i? Or just continue?
                        # Let's just break the inner _ loop and try next i
                    
                    # Also try moving i to a line defined by one other point and a "grid direction"?
                    # No, let's stick to snapping to existing point pairs first.
                    # If no improvement from snapping, try a small random jump?
                    # The current code does that. Let's add a small random jitter if no snap works.
                    
                # If we didn't improve in the snap attempts, try a small random move
                # to escape local minima
                if not improved:
                    # Small random move for point i
                    nx = xi + random.randint(-2, 2)
                    ny = yi + random.randint(-2, 2)
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    if any(nx == p[0] and ny == p[1] for idx, p in enumerate(current_points) if idx != i):
                        continue
                    if nx == xi and ny == yi:
                        continue
                        
                    new_points = [p[:] for p in current_points]
                    new_points[i] = [nx, ny]
                    if not is_valid(new_points):
                        continue
                    new_score = count_ordinary_lines(new_points)
                    if new_score < current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            break
            
    if best_points is None:
        # Fallback to a known valid configuration
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END
