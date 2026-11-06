"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            
            if dx == 0 and dy == 0:
                continue
                
            g = gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Line equation: dy_n * x - dx_n * y = c
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
            
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def _local_search(points, max_iters=10000, deadline=None):
    """Local search: move one point at a time to reduce ordinary lines."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        idx = random.randint(0, n - 1)
        xi, yi = best[idx]
        
        # Try a few random moves
        for _ in range(10):
            # Move within a small range
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
                
            new_pt = [xi + dx, yi + dy]
            
            # Check collision
            valid = True
            for j in range(n):
                if j != idx and best[j] == new_pt:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check bounds
            if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                continue
                
            old_pt = best[idx]
            best[idx] = new_pt
            cnt = _count_ordinary_lines(best)
            best[idx] = old_pt
            
            if cnt < best_cnt:
                best[idx] = new_pt
                best_cnt = cnt
                improved = True
                break
                
        if not improved:
             # Try a larger jump
             dx = random.randint(-50, 50)
             dy = random.randint(-50, 50)
             new_pt = [xi + dx, yi + dy]
             valid = True
             for j in range(n):
                 if j != idx and best[j] == new_pt:
                     valid = False
                     break
             if valid and abs(new_pt[0]) <= 10**9 and abs(new_pt[1]) <= 10**9:
                 old_pt = best[idx]
                 best[idx] = new_pt
                 cnt = _count_ordinary_lines(best)
                 best[idx] = old_pt
                 if cnt < best_cnt:
                     best[idx] = new_pt
                     best_cnt = cnt
                     
    return best, best_cnt

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    random.seed(42)
    
    # Generate diverse initial configurations
    seeds = []
    
    # 3x3 grid + 3 random
    grid_3x3 = [[x, y] for x in range(3) for y in range(3)]
    for _ in range(5):
        pts = grid_3x3[:]
        while len(pts) < 12:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)
        
    # 4x3 grid
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    seeds.append(grid_4x3)
    
    # 3x4 grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    seeds.append(grid_3x4)
    
    # Random
    for _ in range(5):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-20, 20), random.randint(-20, 20)]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)
        
    best_pts = None
    best_cnt = float('inf')
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Evaluate seed
        cnt = _count_ordinary_lines(seed)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in seed]
            
        # Local search
        pts, cnt = _local_search(seed, max_iters=2000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts
            
    # Final refinement
    if best_pts and time.time() < deadline - 10:
        pts, cnt = _local_search(best_pts, max_iters=10000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts
            
    if best_pts is None:
        best_pts = grid_4x3
        
    # Ensure distinctness and bounds
    seen = set()
    final = []
    for p in best_pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
            
    while len(final) < 12:
        p = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
            
    return final
# EVOLVE-BLOCK-END
