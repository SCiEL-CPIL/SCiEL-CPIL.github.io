import time
import random
from math import gcd
from collections import defaultdict
import numpy as np

def count_exact_four_lines_fast(points):
    """Count lines containing exactly 4 of the 16 points. Returns (four_count, five_or_more_count)."""
    n = len(points)
    if n < 4:
        return 0, 0
    
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                c = dy * x1 - dx * y1
            else:
                c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    five_or_more = 0
    for key, pair_count in line_counts.items():
        if pair_count == 6:  # C(4,2)
            four_count += 1
        elif pair_count >= 10:  # C(5,2)
            five_or_more += 1
    return four_count, five_or_more

def evaluate(points):
    """Score: number of 4-point lines, penalty if any 5+ collinear."""
    four, five = count_exact_four_lines_fast(points)
    if five > 0:
        return -1000 + four  # heavy penalty
    return four

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    random.seed(12345)
    
    best_score = -1
    best_points = None
    
    # Phase 1: Try structured configurations
    structured = []
    
    # 4x4 grid at various positions and scales
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            for sx in [1, 2, 3]:
                for sy in [1, 2, 3]:
                    pts = [[ox + sx*x, oy + sy*y] for x in range(4) for y in range(4)]
                    if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                        if len(set(tuple(p) for p in pts)) == 16:
                            structured.append(pts)
    
    # Add some perturbed grids
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    for _ in range(200):
        pts = [p[:] for p in base_grid]
        for i in range(16):
            if random.random() < 0.3:
                pts[i][0] += random.randint(-3, 3)
                pts[i][1] += random.randint(-3, 3)
        if len(set(tuple(p) for p in pts)) == 16:
            structured.append(pts)
    
    for pts in structured:
        if time.time() > deadline:
            break
        s = evaluate(pts)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in pts]
    
    # Phase 2: Simulated annealing
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = evaluate(best_points)
    
    cur_points = [p[:] for p in best_points]
    cur_score = best_score
    
    # Simulated annealing parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    max_iter = 50000
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a point to move
        idx = random.randint(0, 15)
        
        # Choose a move
        if random.random() < 0.7:
            # Small move
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
        else:
            # Larger move
            dx = random.randint(-30, 30)
            dy = random.randint(-30, 30)
        
        if dx == 0 and dy == 0:
            continue
        
        new_pts = [p[:] for p in cur_points]
        new_pts[idx] = [cur_points[idx][0] + dx, cur_points[idx][1] + dy]
        
        # Check bounds and distinctness
        if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
            continue
        if len(set(tuple(p) for p in new_pts)) != 16:
            continue
        
        new_score = evaluate(new_pts)
        delta = new_score - cur_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0:
            cur_points = new_pts
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_points = [p[:] for p in cur_points]
        else:
            # Metropolis criterion
            if random.random() < pow(2.718281828, delta / T):
                cur_points = new_pts
                cur_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Phase 3: Local refinement from best
    cur_points = [p[:] for p in best_points]
    cur_score = best_score
    
    for step in range(30000):
        if time.time() > deadline:
            break
        
        improved = False
        order = list(range(16))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in cur_points]
                    new_pts[idx] = [cur_points[idx][0] + dx, cur_points[idx][1] + dy]
                    if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                        continue
                    if len(set(tuple(p) for p in new_pts)) != 16:
                        continue
                    s = evaluate(new_pts)
                    if s > cur_score:
                        cur_points = new_pts
                        cur_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a random perturbation to escape
            if random.random() < 0.3:
                for _ in range(random.randint(1, 4)):
                    idx = random.randint(0, 15)
                    cur_points[idx][0] += random.randint(-15, 15)
                    cur_points[idx][1] += random.randint(-15, 15)
                    if abs(cur_points[idx][0]) > 1000 or abs(cur_points[idx][1]) > 1000:
                        cur_points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                if len(set(tuple(p) for p in cur_points)) == 16:
                    cur_score = evaluate(cur_points)
            else:
                break
        
        if cur_score > best_score:
            best_score = cur_score
            best_points = [p[:] for p in cur_points]
    
    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final)
    four, five = count_exact_four_lines_fast(final)
    assert five == 0, f"Five collinear found: {five}"
    
    return final
