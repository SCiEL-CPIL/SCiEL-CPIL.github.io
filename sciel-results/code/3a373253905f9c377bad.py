import random
import math
import time
import sys
from fractions import Fraction

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
    
    # Normalize direction vector
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None # Should not happen for distinct points
    a = dy // g
    b = -dx // g
    
    # Normalize sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        
    # Calculate c such that ax + by + c = 0
    c = -(a * x1 + b * y1)
    
    return (a, b, c)

def _evaluate(points):
    """
    Returns (score, is_valid).
    score: number of lines with exactly 4 points.
    is_valid: False if any line has >= 5 points.
    """
    n = len(points)
    
    unique_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key:
                unique_lines.add(key)
                
    score = 0
    is_valid = True
    
    for key in unique_lines:
        a, b, c = key
        count = 0
        for p in points:
            x, y = p
            if a * x + b * y + c == 0:
                count += 1
        if count >= 5:
            is_valid = False
            return 0, False
        elif count == 4:
            score += 1
            
    return score, True

def _try_local_search(points, deadline, max_steps=50000):
    """Perform local search starting from given points."""
    current_score, _ = _evaluate(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    # Precompute all pair indices for efficiency
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    step = 0
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # Choose a random point to move
        idx = random.randint(0, 15)
        
        # Generate candidate new positions
        candidates = []
        
        # 1. Small perturbations of current position
        cx, cy = points[idx]
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    # Check if position is free
                    occupied = False
                    for i, p in enumerate(points):
                        if i != idx and p[0] == nx and p[1] == ny:
                            occupied = True
                            break
                    if not occupied:
                        candidates.append([nx, ny])
        
        # 2. Random positions in a moderate range
        for _ in range(5):
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            occupied = False
            for i, p in enumerate(points):
                if i != idx and p[0] == nx and p[1] == ny:
                    occupied = True
                    break
            if not occupied:
                candidates.append([nx, ny])
        
        # 3. Try to align with existing lines (move to a point that creates new 4-lines)
        # For each line with exactly 3 points, try adding this point to make 4
        line_map = {}
        for i, j in pairs:
            key = _line_key(points[i], points[j])
            if key:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        
        for key, pts_on_line in line_map.items():
            if len(pts_on_line) == 3 and idx not in pts_on_line:
                # Find a point on this line that's not one of the 3
                a, b, c = key
                # Try a few points on this line
                for t in range(-10, 11):
                    # ax + by + c = 0, so if a != 0: x = -(by + c) / a
                    # We need integer solutions
                    if a != 0:
                        # by + c must be divisible by a
                        for y in range(-20, 21):
                            val = b * y + c
                            if val % a == 0:
                                x = -val // a
                                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                    # Check if this point is free and not one of the 3
                                    if (x, y) not in [(points[i][0], points[i][1]) for i in pts_on_line]:
                                        occupied = False
                                        for i, p in enumerate(points):
                                            if i != idx and p[0] == x and p[1] == y:
                                                occupied = True
                                                break
                                        if not occupied:
                                            candidates.append([x, y])
                    elif b != 0:
                        # ax + c must be divisible by b
                        for x in range(-20, 21):
                            val = a * x + c
                            if val % b == 0:
                                y = -val // b
                                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                    if (x, y) not in [(points[i][0], points[i][1]) for i in pts_on_line]:
                                        occupied = False
                                        for i, p in enumerate(points):
                                            if i != idx and p[0] == x and p[1] == y:
                                                occupied = True
                                                break
                                        if not occupied:
                                            candidates.append([x, y])
        
        if not candidates:
            continue
            
        # Pick a random candidate
        new_pos = random.choice(candidates)
        new_points = [p[:] for p in points]
        new_points[idx] = new_pos
        
        new_score, is_valid = _evaluate(new_points)
        
        if is_valid and new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        elif is_valid and new_score == current_score:
            # Accept equal-score moves to explore
            if random.random() < 0.3:
                points = new_points
        # If invalid or worse, reject
        
    return best_points, best_score

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Try multiple initial configurations
    initial_configs = [
        # 4x4 grid
        [[x, y] for x in range(4) for y in range(4)],
        # 4x4 grid shifted
        [[x + 1, y + 1] for x in range(4) for y in range(4)],
        # 4x4 grid scaled
        [[x * 2, y * 2] for x in range(4) for y in range(4)],
        # 4x4 grid with different spacing
        [[x * 3, y * 3] for x in range(4) for y in range(4)],
        # 4x4 grid centered
        [[x - 1, y - 1] for x in range(4) for y in range(4)],
    ]
    
    best_score = 0
    best_points = None
    
    # Evaluate each initial config and run local search on the best ones
    for config in initial_configs:
        if time.time() >= deadline - 20:
            break
            
        score, _ = _evaluate(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
        
        # Run local search if we have time
        if time.time() < deadline - 15:
            search_deadline = min(deadline - 5, time.time() + 15)
            improved_points, improved_score = _try_local_search(config, search_deadline, max_steps=20000)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Final polish: try to improve the best result further
    if best_points and time.time() < deadline - 5:
        final_deadline = deadline - 2
        improved_points, improved_score = _try_local_search(best_points, final_deadline, max_steps=30000)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    return best_points

# EVOLVE-BLOCK-END
