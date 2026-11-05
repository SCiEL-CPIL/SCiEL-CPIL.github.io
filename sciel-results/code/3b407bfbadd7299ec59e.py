"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
import numpy as np

def _gcd2(x, y):
    x, y = abs(x), abs(y)
    while y:
        x, y = y, x % y
    return x

def _gcd3(a, b, c):
    return _gcd2(_gcd2(a, b), c)

def _fast_score(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Returns (score, is_valid).
    """
    n = len(points)
    if n < 2:
        return 0, True
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            g = _gcd3(a, b, c)
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_lines = 0
    for count in line_counts.values():
        val = 1 + 8 * count
        root = math.isqrt(val)
        if root * root != val:
            continue
        k = (1 + root) // 2
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0, False
            
    return four_lines, True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Start with 4x4 grid: known to have 8 four-point lines
    base_points = [[i, j] for i in range(4) for j in range(4)]
    
    best_points = [p[:] for p in base_points]
    best_score, _ = _fast_score(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    n = 16
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        i = random.randint(0, n - 1)
        old_x, old_y = current_points[i]
        
        # Generate new position - try various move sizes
        r = random.random()
        if r < 0.5:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
        elif r < 0.8:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
        else:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        # Check distinctness
        dup = False
        for j in range(n):
            if i != j and current_points[j][0] == new_x and current_points[j][1] == new_y:
                dup = True
                break
        if dup:
            continue
        
        current_points[i] = [new_x, new_y]
        new_score, valid = _fast_score(current_points)
        
        if not valid:
            current_points[i] = [old_x, old_y]
            continue
        
        # Simulated annealing acceptance
        delta = new_score - current_score
        if delta > 0 or (temperature > 0.1 and random.random() < math.exp(delta / temperature)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[i] = [old_x, old_y]
        
        temperature *= cooling
        
        # Periodic restart if stuck
        if temperature < 0.1:
            temperature = 5.0
            for _ in range(10):
                current_points = [p[:] for p in best_points]
                for _ in range(5):
                    idx = random.randint(0, n - 1)
                    ox, oy = current_points[idx]
                    nx = ox + random.randint(-5, 5)
                    ny = oy + random.randint(-5, 5)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    dup = False
                    for j in range(n):
                        if idx != j and current_points[j][0] == nx and current_points[j][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    current_points[idx] = [nx, ny]
                    s, v = _fast_score(current_points)
                    if v and s > current_score:
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_points = [p[:] for p in current_points]
                    else:
                        current_points[idx] = [ox, oy]
        
        # If we hit the theoretical upper bound, stop
        if best_score >= 15:
            break
    
    # Final validation
    final_score, valid = _fast_score(best_points)
    if not valid or final_score == 0:
        best_points = [[i, j] for i in range(4) for j in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
