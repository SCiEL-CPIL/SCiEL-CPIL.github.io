"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def count_ordinary_lines(points):
    """
    Count lines containing exactly 2 of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None
        
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        g = math.gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
            
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def point_on_line(p, line_key):
        if line_key is None:
            return False
        dx, dy, c = line_key
        x, y = p
        return dy * x - dx * y == c

    unique_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is not None:
                unique_lines.add(key)
                
    ordinary_count = 0
    for line_key in unique_lines:
        count = 0
        for p in points:
            if point_on_line(p, line_key):
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(points, count):
        nonlocal best_points, best_count
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in points]
            # If we hit the theoretical minimum, we can stop early
            if count == 6:
                return True
        return False

    # 1. Baseline: 4x3 Grid
    grid_points = [[x, y] for x in range(4) for y in range(3)]
    grid_count = count_ordinary_lines(grid_points)
    if update_best(grid_points, grid_count):
        return best_points

    # 2. Line-Snapping Simulated Annealing
    # This strategy specifically tries to create 3-point lines by moving points onto existing lines.
    
    def snap_move(points, i, max_dist=5):
        """
        Generate a move that places point i on a line defined by two other points j, k.
        Returns a list of candidate new positions for point i.
        """
        candidates = []
        n = len(points)
        x_i, y_i = points[i]
        
        # Find all lines defined by pairs of other points
        for j in range(n):
            if j == i: continue
            for k in range(j+1, n):
                if k == i: continue
                
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                
                dx = x_k - x_j
                dy = y_k - y_j
                
                if dx == 0 and dy == 0: continue
                
                # Normalize direction vector for step
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                step_x = dx // g
                step_y = dy // g
                
                # Line equation: dy*x - dx*y = c
                c = dy * x_j - dx * y_j
                
                # We want to find integer (x, y) close to (x_i, y_i) such that dy*x - dx*y = c
                # and the point is not j or k.
                
                # Parametric form: x = x_j + t*step_x, y = y_j + t*step_y
                # We need to find t such that (x_j + t*step_x, y_j + t*step_y) is close to (x_i, y_i)
                # Minimize (x_j + t*step_x - x_i)^2 + (y_j + t*step_y - y_i)^2
                # Let u = step_x, v = step_y.
                # f(t) = (u*t + (x_j - x_i))^2 + (v*t + (y_j - y_i))^2
                # f'(t) = 2u(u*t + x_j - x_i) + 2v(v*t + y_j - y_i) = 0
                # t(u^2 + v^2) = -(u(x_j - x_i) + v(y_j - y_i))
                # t = - (u*(x_j - x_i) + v*(y_j - y_i)) / (u^2 + v^2)
                
                num = -(step_x * (x_j - x_i) + step_y * (y_j - y_i))
                den = step_x**2 + step_y**2
                
                if den == 0: continue
                
                t_float = num / den
                t_candidates = [int(t_float), int(t_float) + 1, int(t_float) - 1]
                
                for t in t_candidates:
                    nx = x_j + t * step_x
                    ny = y_j + t * step_y
                    
                    # Check if it's actually on the line (integer arithmetic check)
                    if dy * nx - dx * ny != c:
                        continue
                    
                    # Check distance
                    dist = abs(nx - x_i) + abs(ny - y_i)
                    if dist > max_dist:
                        continue
                    
                    # Check bounds
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    
                    # Check if it collides with j or k
                    if (nx == x_j and ny == y_j) or (nx == x_k and ny == y_k):
                        continue
                    
                    candidates.append([nx, ny])
        
        # Remove duplicates
        unique_candidates = []
        seen = set()
        for c in candidates:
            key = (c[0], c[1])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        return unique_candidates

    def simulated_annealing(initial_points, max_iters=50000):
        pts = [p[:] for p in initial_points]
        current_count = count_ordinary_lines(pts)
        best_local_pts = [p[:] for p in pts]
        best_local_count = current_count
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if T < min_T:
                break
            
            # Choose a move
            move_type = random.random()
            i = random.randint(0, 11)
            old_pos = pts[i][:]
            new_pos = None
            
            if move_type < 0.5:
                # Shift move
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0: continue
                new_pos = [old_pos[0] + dx, old_pos[1] + dy]
                if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
                    continue
            elif move_type < 0.9:
                # Snap move
                candidates = snap_move(pts, i, max_dist=5)
                if candidates:
                    new_pos = random.choice(candidates)
                else:
                    continue
            else:
                # Jump move
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_pos = [old_pos[0] + dx, old_pos[1] + dy]
                if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
                    continue
            
            # Check validity (distinctness)
            # Temporarily place new_pos
            pts[i] = new_pos
            if len(set(tuple(p) for p in pts)) < 12:
                pts[i] = old_pos
                continue
            
            new_count = count_ordinary_lines(pts)
            delta = new_count - current_count
            
            # Metropolis criterion
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_count = new_count
                if current_count < best_local_count:
                    best_local_count = current_count
                    best_local_pts = [p[:] for p in pts]
                    if update_best(best_local_pts, best_local_count):
                        return best_local_pts
            else:
                pts[i] = old_pos
            
            T *= cooling
        
        return best_local_pts, best_local_count

    random.seed(42)
    
    # Multiple restarts with different initial configurations
    # 1. Grid
    # 2. Staircase
    # 3. Random small box
    
    initial_configs = []
    
    # Grid 4x3
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Staircase / Zigzag
    # Try to make many 3-point lines
    # e.g. (0,0), (1,1), (2,2), (0,1), (1,2), (2,3) ...
    init_stair = []
    for i in range(6):
        init_stair.append([i, i])
        init_stair.append([i, i+1])
    # Ensure distinct
    init_stair = list(set(tuple(p) for p in init_stair))[:12]
    if len(init_stair) == 12:
        initial_configs.append([list(p) for p in init_stair])
    
    # Random
    for _ in range(5):
        pts = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
        if len(set(tuple(p) for p in pts)) == 12:
            initial_configs.append(pts)
    
    for config in initial_configs:
        if time.time() > deadline - 10.0:
            break
        improved_pts, improved_cnt = simulated_annealing(config)
        if update_best(improved_pts, improved_cnt):
            if best_count == 6:
                break

    if best_points is None:
        return grid_points
    
    return best_points

# EVOLVE-BLOCK-END
