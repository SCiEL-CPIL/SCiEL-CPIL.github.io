import time
import math
import random
import sys

def _clamp(v):
    return -1000 if v < -1000 else (1000 if v > 1000 else v)

def _count_lines(pts):
    """
    Returns (n4, n5plus) where n4 is number of lines with exactly 4 points,
    and n5plus is number of lines with 5 or more points.
    """
    lines = {}
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            # Line equation: A*x + B*y + C = 0
            # A = dy, B = -dx
            A = dy
            B = -dx
            C = dy * xi - dx * yi
            
            # Normalize A, B, C
            gg = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if gg:
                A //= gg
                B //= gg
                C //= gg
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key in lines:
                lines[key].add(i)
                lines[key].add(j)
            else:
                lines[key] = {i, j}
    
    n4 = 0
    n5plus = 0
    for s in lines.values():
        c = len(s)
        if c == 4:
            n4 += 1
        elif c >= 5:
            n5plus += 1
            
    return n4, n5plus

def _score(pts, penalty_weight=100):
    """
    Objective function.
    We want to maximize n4 while minimizing n5plus.
    """
    n4, n5plus = _count_lines(pts)
    return n4 - penalty_weight * n5plus

def _valid(pts):
    if len(pts) != 16:
        return False
    if len(set(tuple(p) for p in pts)) != 16:
        return False
    for x, y in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, n5plus = _count_lines(pts)
    return n5plus == 0

def _make_move(pts, rng, grid_limit=20):
    """
    Generate a candidate by perturbing one or two points.
    """
    cand = [list(p) for p in pts]
    strategy = rng.random()
    
    if strategy < 0.7:
        # Perturb one point
        idx = rng.randrange(16)
        # Try to stay close to current position
        cur_x, cur_y = cand[idx]
        # Random step size
        step = rng.randint(1, 5)
        cand[idx][0] = _clamp(cur_x + rng.randint(-step, step))
        cand[idx][1] = _clamp(cur_y + rng.randint(-step, step))
    elif strategy < 0.9:
        # Perturb two points
        i1, i2 = rng.sample(range(16), 2)
        for idx in [i1, i2]:
            cur_x, cur_y = cand[idx]
            step = rng.randint(1, 5)
            cand[idx][0] = _clamp(cur_x + rng.randint(-step, step))
            cand[idx][1] = _clamp(cur_y + rng.randint(-step, step))
    else:
        # Swap two points' coordinates (if they are distinct)
        i1, i2 = rng.sample(range(16), 2)
        cand[i1], cand[i2] = cand[i2][:], cand[i1][:]
        
    # Ensure distinctness? The score function handles it via penalty if we add it, 
    # but _count_lines might behave weirdly with duplicates.
    # Let's check duplicates and resolve by moving one.
    seen = set()
    for i in range(16):
        p = tuple(cand[i])
        if p in seen:
            # Move point i randomly
            cand[i] = [rng.randint(-100, 100), rng.randint(-100, 100)]
            # Re-check just in case
            # Simple retry loop
            while tuple(cand[i]) in seen:
                cand[i] = [rng.randint(-100, 100), rng.randint(-100, 100)]
        seen.add(tuple(cand[i]))
        
    return cand

def _solve_single(start, time_limit, rng):
    cur = [list(p) for p in start]
    cur_score = _score(cur)
    best = [list(p) for p in cur]
    best_score = cur_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    t_start = time.time()
    
    while time.time() - t_start < time_limit:
        cand = _make_move(cur, rng)
        cand_score = _score(cand)
        
        delta = cand_score - cur_score
        
        if delta > 0 or rng.random() < math.exp(delta / T):
            cur = cand
            cur_score = cand_score
            
            if cur_score > best_score:
                best = [list(p) for p in cur]
                best_score = cur_score
        
        T = max(T_min, T * cooling)
        
        # Occasionally reset T to escape deep local minima?
        # Or use ILS: if stuck, jump.
        # Simple ILS: every 5000 steps, if no improvement, jump
        # We don't track steps easily here, but we can use time.
        
    return best, best_score

def construct():
    t0 = time.time()
    B = 120.0 # Safety margin
    
    rng = random.Random(20240607)
    
    # Generate diverse seeds
    seeds = []
    
    # 1. 4x4 Grid
    seeds.append([[x, y] for x in range(4) for y in range(4)])
    
    # 2. Random points in 7x7 grid
    for _ in range(5):
        pts = []
        seen = set()
        while len(pts) < 16:
            p = (rng.randint(-3, 3), rng.randint(-3, 3))
            if p not in seen:
                seen.add(p)
                pts.append(list(p))
        seeds.append(pts)
        
    # 3. Random points in 11x11 grid
    for _ in range(5):
        pts = []
        seen = set()
        while len(pts) < 16:
            p = (rng.randint(-5, 5), rng.randint(-5, 5))
            if p not in seen:
                seen.add(p)
                pts.append(list(p))
        seeds.append(pts)
        
    # 4. Random points in 21x21 grid
    for _ in range(5):
        pts = []
        seen = set()
        while len(pts) < 16:
            p = (rng.randint(-10, 10), rng.randint(-10, 10))
            if p not in seen:
                seen.add(p)
                pts.append(list(p))
        seeds.append(pts)

    best_global = None
    best_global_score = -1000
    
    n_seeds = len(seeds)
    time_per_seed = (B * 0.8) / n_seeds
    
    for i, seed in enumerate(seeds):
        if time.time() - t0 > B * 0.8:
            break
        end_time = t0 + (i + 1) * time_per_seed
        # Pass time limit relative to start
        res, score = _solve_single(seed, time_per_seed, rng)
        
        if score > best_global_score:
            best_global_score = score
            best_global = res
            
    # Final polish on the best solution found
    if best_global is not None:
        remaining_time = B - (time.time() - t0)
        if remaining_time > 5:
            # Run a quick polish with lower temperature
            # Reuse _solve_single with a smaller budget
            best_global, best_global_score = _solve_single(best_global, remaining_time, rng)
            
    # Fallback if no valid solution found
    if best_global is None or not _valid(best_global):
        # Return the 4x4 grid which is always valid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_global

