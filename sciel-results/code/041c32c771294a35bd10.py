import random
import math
import time
import sys

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    a = dy // g
    b = -dx // g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def _evaluate(points):
    """Returns (score, is_valid). score = #lines with exactly 4 pts, is_valid = no 5+ collinear."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
    
    score = 0
    is_valid = True
    for key, pts in line_map.items():
        cnt = len(pts)
        if cnt >= 5:
            return 0, False
        elif cnt == 4:
            score += 1
    return score, True

def _evaluate_fast(points, moved_idx, old_pos, new_pos):
    """Incremental evaluation: only recompute lines affected by moving one point."""
    n = len(points)
    # Build line map excluding the moved point
    line_map = {}
    for i in range(n):
        if i == moved_idx:
            continue
        for j in range(i + 1, n):
            if j == moved_idx:
                continue
            key = _line_key(points[i], points[j])
            if key is not None:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
    
    # Add lines involving the moved point at its NEW position
    for j in range(n):
        if j == moved_idx:
            continue
        key = _line_key(new_pos, points[j])
        if key is not None:
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(moved_idx)
            line_map[key].add(j)
    
    score = 0
    is_valid = True
    for key, pts in line_map.items():
        cnt = len(pts)
        if cnt >= 5:
            return 0, False
        elif cnt == 4:
            score += 1
    return score, True

def _local_search(points, deadline, max_steps=200000):
    """Hill-climbing local search with targeted and random moves."""
    current_score, _ = _evaluate(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    no_improve = 0
    max_no_improve = 5000
    
    step = 0
    while time.time() < deadline and step < max_steps and no_improve < max_no_improve:
        step += 1
        idx = random.randint(0, 15)
        cx, cy = points[idx]
        
        # Generate candidate positions
        candidates = []
        
        # 1. Small perturbations
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    occupied = False
                    for i, p in enumerate(points):
                        if i != idx and p[0] == nx and p[1] == ny:
                            occupied = True
                            break
                    if not occupied:
                        candidates.append((nx, ny))
        
        # 2. Try to complete 3-point lines: find lines with exactly 3 pts (not including idx)
        # and place idx on that line
        line_map = {}
        for i in range(16):
            if i == idx:
                continue
            for j in range(i + 1, 16):
                if j == idx:
                    continue
                key = _line_key(points[i], points[j])
                if key is not None:
                    if key not in line_map:
                        line_map[key] = set()
                    line_map[key].add(i)
                    line_map[key].add(j)
        
        for key, pts_on_line in line_map.items():
            if len(pts_on_line) == 3:
                a, b, c = key
                # Find integer points on this line within bounds
                # ax + by + c = 0
                if a != 0:
                    # x = -(by + c) / a
                    # Try y values near existing points
                    y_range = range(-20, 21)
                    for y in y_range:
                        val = b * y + c
                        if val % a == 0:
                            x = -val // a
                            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                if not any(points[i][0] == x and points[i][1] == y for i in range(16) if i != idx):
                                    candidates.append((x, y))
                elif b != 0:
                    x_range = range(-20, 21)
                    for x in x_range:
                        val = a * x + c
                        if val % b == 0:
                            y = -val // b
                            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                if not any(points[i][0] == x and points[i][1] == y for i in range(16) if i != idx):
                                    candidates.append((x, y))
        
        # 3. A few random positions in moderate range
        for _ in range(3):
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            if not any(points[i][0] == nx and points[i][1] == ny for i in range(16) if i != idx):
                candidates.append((nx, ny))
        
        if not candidates:
            no_improve += 1
            continue
        
        # Try up to 5 random candidates
        improved = False
        random.shuffle(candidates)
        for nx, ny in candidates[:5]:
            new_points = [p[:] for p in points]
            new_points[idx] = [nx, ny]
            new_score, is_valid = _evaluate_fast(new_points, idx, (cx, cy), (nx, ny))
            if is_valid and new_score > current_score:
                points = new_points
                current_score = new_score
                no_improve = 0
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                break
        
        if not improved:
            no_improve += 1
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Multiple initial configurations
    initial_configs = [
        [[x, y] for x in range(4) for y in range(4)],
        [[x * 2, y * 2] for x in range(4) for y in range(4)],
        [[x * 3, y * 3] for x in range(4) for y in range(4)],
        [[x, y] for x in range(5) for y in range(4)][:16],
        [[x * 2, y] for x in range(4) for y in range(4)],
        [[x, y * 2] for x in range(4) for y in range(4)],
    ]
    
    best_score = 0
    best_points = None
    
    for config in initial_configs:
        if time.time() > deadline - 10:
            break
        score, valid = _evaluate(config)
        if not valid:
            continue
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score, _ = _evaluate(best_points)
    
    # Run local search with time budget
    search_deadline = deadline - 5.0
    best_points, best_score = _local_search(best_points, search_deadline)
    
    # If time remains, try another restart from a different config
    if time.time() < deadline - 15:
        for config in initial_configs:
            if time.time() > deadline - 10:
                break
            score, valid = _evaluate(config)
            if not valid or score < best_score:
                continue
            # Run brief local search
            sub_deadline = min(deadline - 5.0, time.time() + 10.0)
            pts, sc = _local_search(config, sub_deadline, max_steps=50000)
            if sc > best_score:
                best_score = sc
                best_points = pts
    
    return best_points

# EVOLVE-BLOCK-END
