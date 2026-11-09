import math
import time
import random
from math import gcd

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    line_pair_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
    return sum(1 for c in line_pair_counts.values() if c == 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    best_points = None
    best_count = float('inf')
    
    # Good initial configurations
    configs = [
        [[x, y] for x in range(4) for y in range(3)],  # 4x3
        [[x, y] for x in range(3) for y in range(4)],  # 3x4
        [[x, y] for x in range(6) for y in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]][:12],  # Not great
        [[x, x*x] for x in range(12)],  # Parabola
        [[0,0],[1,0],[-1,0],[0,1],[0,-1],[2,0],[-2,0],[0,2],[0,-2],[1,1],[-1,-1],[1,-1]],
        [[x, y] for x in range(-2, 4) for y in range(3)][:12],  # 6x2
        [[x, y] for x in range(2) for y in range(6)],  # 2x6
    ]
    
    random.seed(42)
    for _ in range(10):
        pts = set()
        while len(pts) < 12:
            pts.add((random.randint(-10, 10), random.randint(-10, 10)))
        configs.append([list(p) for p in pts])
    
    for init_pts in configs:
        if time.time() > deadline:
            break
        if len(set(tuple(p) for p in init_pts)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in init_pts):
            continue
        
        curr_pts = [list(p) for p in init_pts]
        curr_count = count_ordinary_lines(curr_pts)
        
        if curr_count < best_count:
            best_count = curr_count
            best_points = [list(p) for p in curr_pts]
        
        T = 1.0
        cooling_rate = 0.9995
        steps = 0
        max_steps = 8000
        
        while steps < max_steps and time.time() < deadline:
            idx = random.randint(0, 11)
            old_x, old_y = curr_pts[idx]
            
            if random.random() < 0.6:
                new_x = old_x + random.randint(-2, 2)
                new_y = old_y + random.randint(-2, 2)
            else:
                new_x = old_x + random.randint(-8, 8)
                new_y = old_y + random.randint(-8, 8)
                
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                steps += 1
                continue
            if any(curr_pts[k] == [new_x, new_y] for k in range(12) if k != idx):
                steps += 1
                continue
                
            curr_pts[idx] = [new_x, new_y]
            new_count = count_ordinary_lines(curr_pts)
            delta = new_count - curr_count
            
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                curr_count = new_count
                if curr_count < best_count:
                    best_count = curr_count
                    best_points = [list(p) for p in curr_pts]
            else:
                curr_pts[idx] = [old_x, old_y]
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
            steps += 1
    
    return best_points
