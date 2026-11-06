import time
import math
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, _gcd(dx, c))
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Try known good configurations
    # 4x3 grid
    base_grid = [[x, y] for x in range(4) for y in range(3)]
    best_points = base_grid[:]
    best_score = _count_ordinary(base_grid)
    
    # Try sheared grids
    for k in range(-5, 6):
        pts = [[x + k * y, y] for x in range(4) for y in range(3)]
        if len(set(map(tuple, pts))) == 12:
            s = _count_ordinary(pts)
            if s < best_score:
                best_score = s
                best_points = pts
    
    # 2. Simulated Annealing
    # Pool of points: 15x15 grid centered at 0
    pool = [(x, y) for x in range(-7, 8) for y in range(-7, 8)]
    pool = [p for p in pool if p not in [tuple(p) for p in best_points]]
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Annealing parameters
    T_start = 10.0
    T_end = 0.01
    iterations = 20000
    
    for i in range(iterations):
        if time.time() > deadline:
            break
            
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        # Generate a move
        move_type = random.random()
        new_points = [p[:] for p in current_points]
        
        if move_type < 0.5:
            # Swap two points
            i, j = random.sample(range(12), 2)
            # Pick a new point for i and a new point for j from pool or swap with existing
            # To maintain diversity, pick from pool
            p_new_i = random.choice(pool)
            p_new_j = random.choice(pool)
            
            if p_new_i == tuple(current_points[i]) or p_new_i == tuple(current_points[j]):
                continue
            if p_new_j == tuple(current_points[i]) or p_new_j == tuple(current_points[j]):
                continue
            if p_new_i == p_new_j:
                continue
            if p_new_i in [tuple(p) for p in current_points] or p_new_j in [tuple(p) for p in current_points]:
                 # Allow swapping with existing points? No, just replacement from pool
                 # If we replace with a point already in the set, we lose distinctness
                 # So we must ensure p_new_i and p_new_j are not in the current set
                 # Unless we are swapping two existing points
                 pass
            
            # Simpler move: Swap two existing points with two pool points
            # Ensure no duplicates
            current_set = set(map(tuple, current_points))
            if p_new_i in current_set or p_new_j in current_set:
                continue
                
            new_points[i] = list(p_new_i)
            new_points[j] = list(p_new_j)
            
        elif move_type < 0.9:
            # Move one point to a random pool point
            i = random.randint(0, 11)
            p_new = random.choice(pool)
            if p_new in set(map(tuple, current_points)):
                continue
            new_points[i] = list(p_new)
            
        else:
            # Swap two existing points (reordering doesn't change geometry, so skip)
            # Instead, try a "line-preserving" move? Hard to implement.
            # Just do another random replacement
            i = random.randint(0, 11)
            p_new = random.choice(pool)
            if p_new in set(map(tuple, current_points)):
                continue
            new_points[i] = list(p_new)
            
        new_score = _count_ordinary(new_points)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                if best_score == 6:
                    return best_points
    
    return best_points
