import numpy as np
import time
import random
from math import gcd

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _generate_line_pool():
    pool = set()
    # Coefficients up to 4 to cover simple slopes
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            # Constant term range
            # Points are expected to be in roughly [-10, 15]
            # A*x + B*y + C = 0 => C = -(A*x + B*y)
            # Max |C| approx 4*15 + 4*15 = 120? 
            # Let's use a safe range [-60, 60]
            for C in range(-60, 61):
                canon = _normalize_line(A, B, C)
                if canon:
                    pool.add(canon)
    return list(pool)

def _make_grid(rows, cols):
    pts = []
    for x in range(cols):
        for y in range(rows):
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def _make_random(n, seed=42):
    rng = random.Random(seed)
    pts = []
    seen = set()
    while len(pts) < n:
        x = rng.randint(-5, 10)
        y = rng.randint(-5, 10)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def _run_search(initial_pts, pool, point_line_map, deadline):
    n_pts = len(initial_pts)
    n_pool = len(pool)
    
    # Precompute line coeffs for vectorization if needed, 
    # but we use sparse updates via point_line_map
    
    pts = initial_pts.copy()
    
    # Init counts
    counts = np.zeros(n_pool, dtype=np.int64)
    for i in range(n_pts):
        x, y = pts[i]
        indices = point_line_map.get((x, y), np.array([], dtype=int))
        if len(indices) > 0:
            counts[indices] += 1
            
    def get_top_k_sum(c, k=27):
        if n_pool < k:
            return int(np.sum(c))
        top_k = np.partition(c, -k)[-k:]
        return int(np.sum(top_k))
        
    current_score = get_top_k_sum(counts)
    best_score = current_score
    best_pts = pts.copy()
    best_counts = counts.copy()
    
    start_time = time.time()
    rng = random.Random(123)
    
    T = 1.0
    T_min = 0.01
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
            
        i = rng.randint(0, n_pts - 1)
        old_x, old_y = pts[i]
        
        # Propose new point
        if rng.random() < 0.8:
            dx = rng.randint(-2, 2)
            dy = rng.randint(-2, 2)
            new_x, new_y = old_x + dx, old_y + dy
        else:
            new_x = rng.randint(-8, 15)
            new_y = rng.randint(-8, 15)
            
        # Check bounds for map (Map covers -10 to 15)
        if new_x < -10 or new_x > 15 or new_y < -10 or new_y > 15:
            continue
            
        # Check distinctness
        is_dup = False
        for k in range(n_pts):
            if k != i and pts[k, 0] == new_x and pts[k, 1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Update counts
        old_indices = point_line_map.get((old_x, old_y), np.array([], dtype=int))
        new_indices = point_line_map.get((new_x, new_y), np.array([], dtype=int))
        
        if len(old_indices) > 0:
            counts[old_indices] -= 1
        if len(new_indices) > 0:
            counts[new_indices] += 1
            
        new_score = get_top_k_sum(counts)
        
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            pts[i] = [new_x, new_y]
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_counts = counts.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            T *= 0.999
            if T < T_min:
                T = T_min
        else:
            # Revert
            if len(old_indices) > 0:
                counts[old_indices] += 1
            if len(new_indices) > 0:
                counts[new_indices] -= 1
            no_improve += 1
            
    # Extract best lines
    top_indices = np.argsort(best_counts)[-27:]
    best_lines = [pool[i] for i in top_indices]
    
    return best_pts, np.array(best_lines, dtype=np.int64), best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580 # 20s margin
    
    pool = _generate_line_pool()
    n_pool = len(pool)
    
    # Precompute point_line_map for the bounding box
    # Box: x in [-10, 15], y in [-10, 15]
    A_pool = np.array([l[0] for l in pool])
    B_pool = np.array([l[1] for l in pool])
    C_pool = np.array([l[2] for l in pool])
    
    point_line_map = {}
    for x in range(-10, 16):
        for y in range(-10, 16):
            vals = A_pool * x + B_pool * y + C_pool
            mask = vals == 0
            indices = np.where(mask)[0]
            point_line_map[(x, y)] = indices
            
    configs = [
        _make_grid(9, 6), # 6x9
        _make_grid(18, 3), # 3x18
        _make_grid(27, 2), # 2x27
        _make_random(54)
    ]
    
    best_overall_pts = None
    best_overall_lines = None
    best_overall_score = -1
    
    time_per_config = (deadline - start_time) / len(configs)
    
    for i, init_pts in enumerate(configs):
        t_end = start_time + time_per_config * (i + 1)
        if i == len(configs) - 1:
            t_end = deadline - 5
            
        try:
            pts, lines, score = _run_search(init_pts, pool, point_line_map, t_end)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts
                best_overall_lines = lines
        except Exception:
            pass
            
    if best_overall_pts is None:
        # Fallback
        best_overall_pts = _make_grid(9, 6)
        # Calculate lines for fallback
        counts = np.zeros(n_pool, dtype=np.int64)
        for i in range(54):
            x, y = best_overall_pts[i]
            indices = point_line_map.get((x, y), np.array([], dtype=int))
            if len(indices) > 0:
                counts[indices] += 1
        top_indices = np.argsort(counts)[-27:]
        best_overall_lines = np.array([pool[i] for i in top_indices], dtype=np.int64)
        
    return best_overall_pts, best_overall_lines
