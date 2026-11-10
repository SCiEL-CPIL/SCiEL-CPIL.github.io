"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def _count_four_lines(points):
    """Count lines with exactly 4 points, and check no 5-collinear."""
    n = len(points)
    if n != 16:
        return -1, False
    
    # Check all distinct pairs to find lines
    # A line is determined by 2 points; count how many points are on it
    lines = {}  # (slope, intercept_normalized) -> set of point indices
    # Use a normalized representation for lines
    # For line through (x1,y1) and (x2,y2):
    #   direction: (dx, dy) = (x2-x1, y2-y1), normalize by gcd
    #   Use (a, b, c) where ax + by + c = 0, normalized
    
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # a*x + b*y + c = 0
        # a = y2 - y1, b = x1 - x2, c = -(a*x1 + b*y1)
        a = y2 - y1
        b = x1 - x2
        c = -(a * x1 + b * y1)
        # Normalize: make gcd of a,b,c = 1, and first non-zero positive
        from math import gcd
        g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
        if g > 0:
            a //= g
            b //= g
            c //= g
        # Make first non-zero coefficient positive
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
        return (a, b, c)
    
    # Map each line to the set of points on it
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = line_key(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
    
    # Now for each line, count how many of our 16 points are on it
    four_count = 0
    five_or_more = False
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt == 4:
            four_count += 1
        elif cnt >= 5:
            five_or_more = True
    
    return four_count, not five_or_more

def _score(points):
    """Return score: four_count if valid, else negative."""
    four_count, valid = _count_four_lines(points)
    if not valid:
        return -1000 - four_count
    return four_count

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    best_score = -1
    best_points = None
    
    # Initial configurations to try
    initial_configs = []
    
    # 4x4 grid
    grid44 = [[x, y] for x in range(4) for y in range(4)]
    initial_configs.append(grid44)
    
    # 4x4 grid shifted
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            cfg = [[x+dx, y+dy] for x in range(4) for y in range(4)]
            initial_configs.append(cfg)
    
    # Some other structured configs
    # 2x8 rectangle
    initial_configs.append([[x, y] for x in range(8) for y in range(2)])
    # 8x2
    initial_configs.append([[x, y] for x in range(2) for y in range(8)])
    
    # Random initial configurations
    for _ in range(20):
        pts = []
        while len(pts) < 16:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        initial_configs.append(pts)
    
    # Evaluate initial configs
    for cfg in initial_configs:
        if time.time() > deadline - 5:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in cfg]
    
    # Local search / simulated annealing
    if best_points is None:
        best_points = [list(p) for p in grid44]
        best_score = _score(best_points)
    
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # Simulated annealing
    temperature = 10.0
    cooling = 0.995
    min_temp = 0.01
    
    while time.time() < deadline - 2 and temperature > min_temp:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Move one point
            idx = random.randint(0, 15)
            old = current_points[idx][:]
            nx = old[0] + random.randint(-10, 10)
            ny = old[1] + random.randint(-10, 10)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            current_points[idx] = [nx, ny]
        elif move_type < 0.8:
            # Swap two points
            i, j = random.sample(range(16), 2)
            current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
        else:
            # Replace one point with a random nearby point
            idx = random.randint(0, 15)
            old = current_points[idx][:]
            nx = old[0] + random.randint(-20, 20)
            ny = old[1] + random.randint(-20, 20)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            current_points[idx] = [nx, ny]
        
        new_score = _score(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
        else:
            # Accept with probability based on temperature
            delta = new_score - current_score
            if delta < 0 and random.random() < pow(2.718281828, delta / max(temperature, 0.001)):
                current_score = new_score
            else:
                # Revert
                if move_type < 0.5 or move_type >= 0.8:
                    current_points[idx] = old
                else:
                    current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
        
        temperature *= cooling
    
    # Final polish: try small perturbations around best
    if time.time() < deadline - 1:
        for _ in range(500):
            if time.time() > deadline - 1:
                break
            trial = [list(p) for p in best_points]
            idx = random.randint(0, 15)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            trial[idx] = [trial[idx][0] + dx, trial[idx][1] + dy]
            if abs(trial[idx][0]) > 1000 or abs(trial[idx][1]) > 1000:
                continue
            s = _score(trial)
            if s > best_score:
                best_score = s
                best_points = trial
    
    # Verify final result
    final_score, valid = _count_four_lines(best_points)
    if not valid or final_score < 0:
        # Fallback to 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
