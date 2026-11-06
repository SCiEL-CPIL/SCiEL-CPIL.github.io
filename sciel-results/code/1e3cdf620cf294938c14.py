import random
import time
import math
from collections import defaultdict

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    # Use local vars for speed
    gcd = math.gcd
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        for j in range(i + 1, n):
            x2, y2 = points[j][0], points[j][1]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Normalize direction
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            
            # Line equation: a*x + b*y + c = 0
            # Normal vector (a, b) = (dy, -dx)
            a = dy
            b = -dx
            c = a * x1 + b * y1
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _make_initial_configs():
    configs = []
    
    # 1. Standard Grids (Baseline)
    configs.append([[x, y] for x in range(4) for y in range(3)])
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2. Diagonal-Shift Grids (Strong Candidate)
    # Structure: Points (x+y, y) for x in 0..2, y in 0..3
    # This creates many diagonal lines.
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x + y, y])
    configs.append(pts_diag)
    
    # Variant: (x, y+x)
    pts_diag2 = []
    for y in range(4):
        for x in range(3):
            pts_diag2.append([x, y + x])
    configs.append(pts_diag2)

    # 3. Hexagonal-like / Staggered Rows
    # Two rows of 6, offset
    pts_hex = []
    for x in range(6):
        pts_hex.append([2 * x, 0])
    for x in range(6):
        pts_hex.append([2 * x + 1, 2])
    configs.append(pts_hex)
    
    # 4. 4x4 Grid minus corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    for corner in [[0,0],[3,0],[0,3],[3,3]]:
        if corner in pts:
            pts.remove(corner)
    configs.append(pts)
    
    # 5. "Near Pencil" variants
    # 3 points on each of 4 lines through origin (scaled)
    # Lines: y=0, x=0, y=x, y=-x
    pts_pencil = []
    for d in [1, 2, 3]:
        pts_pencil.append([d, 0])
        pts_pencil.append([0, d])
        pts_pencil.append([d, d])
        pts_pencil.append([-d, d]) # y = -x
    configs.append(pts_pencil)
    
    # 6. Random structured configs for diversity
    rng = random.Random(12345)
    for _ in range(10):
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-5, 5), rng.randint(-5, 5)])
        if len(set(map(tuple, pts))) == 12:
            configs.append(pts)
            
    # Filter valid configs
    valid_configs = []
    for cfg in configs:
        if len(cfg) != 12:
            continue
        if len(set(map(tuple, cfg))) < 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in cfg):
            continue
        # Check not all collinear
        (x0, y0), (x1, y1), (x2, y2) = cfg[0], cfg[1], cfg[2]
        all_col = True
        for i in range(3, 12):
            xi, yi = cfg[i]
            if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                all_col = False
                break
        if all_col:
            continue
        valid_configs.append(cfg)
        
    return valid_configs

def _find_collinear_targets(points):
    """
    Generate candidate positions for a new point that would be collinear 
    with at least 2 existing points.
    """
    n = len(points)
    targets = set()
    gcd = math.gcd
    
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        for j in range(i + 1, n):
            x2, y2 = points[j][0], points[j][1]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # The line passes through (x1, y1) with direction (dx, dy).
            # Integer points on this line are (x1 + k*dx, y1 + k*dy).
            # We want small k to keep coordinates reasonable and likely to improve density.
            for k in range(-4, 5):
                if k == 0:
                    continue
                nx = x1 + k * dx
                ny = y1 + k * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    # Only add if it's not already a point
                    if [nx, ny] not in points:
                        targets.add((nx, ny))
                        
    return list(targets)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s/150s limit
    
    initial_configs = _make_initial_configs()
    
    best_points = None
    best_score = float('inf')
    
    # Initialize with best of initial configs
    for cfg in initial_configs:
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
        if best_score == 6:
            break
            
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)

    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 2000 # Restart if stuck
    
    while time.time() < deadline:
        # Check time before heavy operations
        if time.time() > deadline - 2.0:
            break
            
        strategy = rng.random()
        
        if strategy < 0.6:
            # Strategy 1: Move a point to a collinear target
            # This is the most promising structural move
            targets = _find_collinear_targets(best_points)
            if targets:
                idx = rng.randint(0, 11)
                target = rng.choice(targets)
                new_p = [target[0], target[1]]
                
                # Ensure distinctness
                if any(new_p == p for p in best_points if p is not best_points[idx]):
                    continue
                if new_p == best_points[idx]:
                    continue
                
                new_points = best_points[:]
                new_points[idx] = new_p
                
                # Quick validity check
                if len(set(map(tuple, new_points))) < 12:
                    continue
                if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in new_points):
                    continue

                score = count_ordinary_lines(new_points)
                if score < best_score:
                    best_score = score
                    best_points = new_points
                    no_improve_count = 0
                    if best_score == 6:
                        break
                else:
                    no_improve_count += 1

        elif strategy < 0.9:
            # Strategy 2: Random local move (Perturbation)
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]
            
            step_size = rng.choice([1, 2, 3, 5, 8, 12, 16, 24])
            dx = rng.randint(-step_size, step_size)
            dy = rng.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue
                
            new_p = [old_p[0] + dx, old_p[1] + dy]
            
            if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
                continue
                
            new_points = best_points[:]
            new_points[idx] = new_p
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                no_improve_count = 0
                if best_score == 6:
                    break
            else:
                no_improve_count += 1

        else:
            # Strategy 3: Swap two points (less effective but helps mixing)
            i, j = rng.sample(range(12), 2)
            new_points = best_points[:]
            new_points[i], new_points[j] = best_points[j][:], best_points[i][:]
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                no_improve_count = 0
                if best_score == 6:
                    break
            else:
                no_improve_count += 1

        # Restart logic
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Pick a random initial config to restart from
            if initial_configs:
                cfg = rng.choice(initial_configs)
                s = count_ordinary_lines(cfg)
                if s < best_score:
                    best_score = s
                    best_points = [p[:] for p in cfg]
                else:
                    # Keep current best but "kick" it by randomizing one point heavily
                    idx = rng.randint(0, 11)
                    best_points[idx] = [rng.randint(-20, 20), rng.randint(-20, 20)]
                    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Best Score: {count_ordinary_lines(pts)}")
    print(f"Points: {pts}")
