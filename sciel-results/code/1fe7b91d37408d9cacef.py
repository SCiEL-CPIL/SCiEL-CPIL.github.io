# EVOLVE-BLOCK-START
import math
import time

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
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
            
            if key not in lines:
                lines[key] = set()
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

def deterministic_improve(points, max_iter=200):
    """
    Deterministically improve the configuration by moving points to positions
    that reduce the number of ordinary lines.
    """
    points = [p[:] for p in points]
    current_score = count_ordinary_lines(points)
    
    for _ in range(max_iter):
        improved = False
        for i in range(12):
            best_score = current_score
            best_pos = None
            px, py = points[i]
            
            # Try nearby positions in a spiral
            for r in range(0, 6):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if r == 0 and dx == 0 and dy == 0:
                            continue
                        if abs(dx) != r and abs(dy) != r:
                            continue
                            
                        nx, ny = px + dx, py + dy
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        
                        # Check collision
                        valid = True
                        for j in range(12):
                            if j != i and points[j][0] == nx and points[j][1] == ny:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_points = [p[:] for p in points]
                        new_points[i] = [nx, ny]
                        
                        if not is_valid(new_points):
                            continue
                            
                        new_score = count_ordinary_lines(new_points)
                        if new_score < best_score:
                            best_score = new_score
                            best_pos = [nx, ny]
                            improved = True
                            
            if best_pos:
                points[i] = best_pos
                current_score = best_score
        
        # Try swaps
        for i in range(12):
            for j in range(i + 1, 12):
                new_points = [p[:] for p in points]
                new_points[i], new_points[j] = new_points[j], new_points[i]
                if is_valid(new_points):
                    new_score = count_ordinary_lines(new_points)
                    if new_score < current_score:
                        points = new_points
                        current_score = new_score
                        improved = True
        
        if not improved:
            break
            
    return points, current_score

def construct():
    best_points = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 110
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. 3x4 Grid
    for x in range(3):
        for y in range(4):
            initial_configs.append([x, y])
            
    # 2. 4x3 Grid
    for x in range(4):
        for y in range(3):
            initial_configs.append([x, y])
            
    # 3. 2x6 Grid
    for x in range(2):
        for y in range(6):
            initial_configs.append([x, y])
            
    # 4. 6x2 Grid
    for x in range(6):
        for y in range(2):
            initial_configs.append([x, y])
            
    # 5. Perturbed 3x4 Grid (small integer shifts)
    base = [[x, y] for x in range(3) for y in range(4)]
    for shift_x in range(-1, 2):
        for shift_y in range(-1, 2):
            config = [[x + shift_x, y + shift_y] for x, y in base]
            # Ensure distinctness and validity
            if len(set(tuple(p) for p in config)) == 12:
                initial_configs.append(config)

    # 6. Random but structured: 2 lines of 6
    for i in range(6):
        initial_configs.append([i, 0])
    for i in range(6):
        initial_configs.append([i, 1])
        
    # 7. Random but structured: 3 lines of 4
    for i in range(4):
        initial_configs.append([i, 0])
    for i in range(4):
        initial_configs.append([i, 1])
    for i in range(4):
        initial_configs.append([i, 2])

    # Deduplicate initial configs
    unique_configs = []
    seen = set()
    for config in initial_configs:
        # Normalize: sort points
        sorted_config = sorted([tuple(p) for p in config])
        key = tuple(sorted_config)
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)

    for config in unique_configs:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Ensure validity
        if not is_valid(config):
            continue
            
        # Deterministic improvement
        improved_points, score = deterministic_improve(config, max_iter=50)
        
        if score < best_score:
            best_score = score
            best_points = improved_points
            
        if best_score == 6:
            break
            
    # If no valid config found or all failed, fallback
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END
