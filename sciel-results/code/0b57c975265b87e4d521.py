import time
import random
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _evaluate(points):
    """
    Returns (is_valid, score).
    is_valid: False if any 5 points are collinear.
    score: Number of lines containing exactly 4 points.
    """
    n = len(points)
    line_pair_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key is not None:
                line_pair_counts[key] += 1
            
    valid = True
    score = 0
    
    for count in line_pair_counts.values():
        # Determine k from count: k*(k-1)/2 = count
        k = -1
        for possible_k in range(2, 17):
            if possible_k * (possible_k - 1) // 2 == count:
                k = possible_k
                break
        
        if k == -1:
            continue
            
        if k >= 5:
            valid = False
            break
        elif k == 4:
            score += 1
            
    if not valid:
        return False, 0
        
    return True, score

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Standard 4x4 Grid
    grid4 = [[x, y] for x in range(4) for y in range(4)]
    configs.append(grid4)
    
    # 2. 5x5 Grid Subset (remove 9 points to get 16)
    grid5 = [[x, y] for x in range(5) for y in range(5)]
    random.seed(123)
    indices = list(range(25))
    random.shuffle(indices)
    keep_indices = indices[:16]
    grid5_subset = [grid5[i] for i in sorted(keep_indices)]
    configs.append(grid5_subset)
    
    # 3. Random points
    random.seed(42)
    pts = set()
    while len(pts) < 16:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        pts.add((x, y))
    configs.append([list(p) for p in pts])
    
    # 4. Perturbed 4x4 Grid
    p_grid = [list(p) for p in grid4]
    random.seed(100)
    for i in range(16):
        if random.random() < 0.5:
            p_grid[i][0] += random.randint(-2, 2)
        if random.random() < 0.5:
            p_grid[i][1] += random.randint(-2, 2)
    distinct_p_grid = []
    for p in p_grid:
        if p not in distinct_p_grid:
            distinct_p_grid.append(p)
        else:
            p[0] += 1
            if p[0] > 1000: p[0] = -1000
            if p in distinct_p_grid:
                p[1] += 1
                if p[1] > 1000: p[1] = -1000
            distinct_p_grid.append(p)
    configs.append(distinct_p_grid)
    
    return configs

def _sa_search(initial_points, deadline):
    """
    Simulated Annealing search.
    Returns best points and score found.
    """
    current = [list(p) for p in initial_points]
    c_valid, c_score = _evaluate(current)
    if not c_valid:
        c_score = -1000
        c_valid = True # Treat as very bad but allow moves
        
    best = [list(p) for p in current]
    best_score = c_score
    
    temp = 10.0
    cooling = 0.999
    no_improve = 0
    max_no_improve = 10000 # Increase to allow more exploration
    
    while time.time() < deadline - 2.0 and no_improve < max_no_improve:
        move_type = random.random()
        cand = [list(p) for p in current]
        
        if move_type < 0.3:
            # 1. Jitter: Small random move
            idx = random.randint(0, 15)
            old_x, old_y = cand[idx]
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(cand) if i != idx):
                continue
            cand[idx] = [new_x, new_y]
            
        elif move_type < 0.6:
            # 2. Align with 2 points: Pick 2 other points, try to place current point on their line
            if len(current) >= 3:
                p1_idx, p2_idx = random.sample(range(16), 2)
                x1, y1 = current[p1_idx]
                x2, y2 = current[p2_idx]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                
                available = [i for i in range(16) if i != p1_idx and i != p2_idx]
                if not available:
                    continue
                idx = random.choice(available)
                
                found = False
                for _ in range(10):
                    t = random.randint(-20, 20)
                    if t == 0: continue
                    new_x = x1 + t * dx
                    new_y = y1 + t * dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(cand) if i != idx):
                        continue
                    cand[idx] = [new_x, new_y]
                    found = True
                    break
                if not found:
                    continue
            else:
                continue
                
        elif move_type < 0.8:
            # 3. Swap two points
            i1, i2 = random.sample(range(16), 2)
            cand[i1], cand[i2] = cand[i2], cand[i1]
        else:
            # 4. Global random move
            idx = random.randint(0, 15)
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(cand) if i != idx):
                continue
            cand[idx] = [new_x, new_y]
        
        cand_valid, cand_score = _evaluate(cand)
        if not cand_valid:
            cand_score = -1000
            
        if cand_score > best_score:
            best = [list(p) for p in cand]
            best_score = cand_score
            no_improve = 0
        else:
            no_improve += 1
            
        if cand_score > c_score:
            current = cand
            c_score = cand_score
        else:
            delta = cand_score - c_score
            if delta < 0:
                prob = pow(2.718281828, delta / max(temp, 0.01))
                if random.random() < prob:
                    current = cand
                    c_score = cand_score
                    
        temp *= cooling
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    inits = _generate_initial_configs()
    
    best_overall = None
    best_score_overall = -1
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        alloc_time = min(deadline - time.time(), 25.0)
        if alloc_time < 5.0:
            break
            
        res_pts, res_score = _sa_search(init, deadline)
        
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_overall = res_pts
            
    if best_overall and time.time() < deadline - 5:
        res_pts, res_score = _sa_search(best_overall, deadline)
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_overall = res_pts
            
    if best_overall is None:
        return inits[0]
        
    return best_overall
