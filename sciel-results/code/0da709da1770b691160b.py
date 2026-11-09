import time
import random
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary(pts):
    """Exact count of lines containing exactly two of the given points."""
    n = len(pts)
    # Map each line to the count of points on it.
    # Key: (dx, dy, c) where line is dy*x - dx*y = c
    # We normalize (dx, dy) to be coprime and canonical direction.
    line_counts = {}
    
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            
            g = _gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
            # Canonical direction: dx > 0 or (dx == 0 and dy > 0)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
                
    ordinary = 0
    for count in line_counts.values():
        if count == 1: # 2 points define 1 pair
            ordinary += 1
            
    return ordinary

def _valid(pts):
    if len(pts) != 12:
        return False
    seen = set()
    for x, y in pts:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # not all collinear
    p0 = pts[0]
    p1 = pts[1]
    for i in range(2, 12):
        p2 = pts[i]
        if (p1[0]-p0[0])*(p2[1]-p0[1]) != (p1[1]-p0[1])*(p2[0]-p0[0]):
            return True
    return False

def _generate_candidates(cur_pts, idx):
    """
    Generate candidates for moving point idx.
    Strategy:
    1. Small random perturbations.
    2. Move to a point on a line defined by two other points.
    3. Move to a point that creates a 3-point line with two existing points.
    """
    cands = set()
    current = cur_pts[idx]
    cx, cy = current
    
    # 1. Random perturbations
    for _ in range(20):
        x = cx + random.randint(-10, 10)
        y = cy + random.randint(-10, 10)
        if abs(x) <= 10**9 and abs(y) <= 10**9:
            cands.add((x, y))
            
    # 2. Line-based moves
    # For every pair of other points (j, k), consider the line passing through them.
    # Try to place point idx on this line at an integer coordinate close to current position or at a "nice" spot.
    for j in range(12):
        if j == idx: continue
        for k in range(j+1, 12):
            if k == idx: continue
            
            x1, y1 = cur_pts[j]
            x2, y2 = cur_pts[k]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0: continue
            
            g = _gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
            # Direction vector (dx, dy) is primitive.
            # Points on the line are (x1 + t*dx, y1 + t*dy) for integer t.
            # We want to find t such that (x1 + t*dx, y1 + t*dy) is close to (cx, cy) or just valid.
            # Estimate t closest to current point:
            # Minimize (x1 + t*dx - cx)^2 + (y1 + t*dy - cy)^2
            # Derivative w.r.t t: dx(x1 + t*dx - cx) + dy(y1 + t*dy - cy) = 0
            # t * (dx^2 + dy^2) = dx*(cx - x1) + dy*(cy - y1)
            
            denom = dx*dx + dy*dy
            num = dx*(cx - x1) + dy*(cy - y1)
            t0 = num / denom
            
            # Check t values around t0
            for t in range(int(t0) - 5, int(t0) + 6):
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                pt = (nx, ny)
                # Avoid duplicates with existing points
                if pt == current:
                    continue
                # Check if pt is already in cur_pts
                # We can just add to set and check validity later, or check here for efficiency
                is_dup = False
                for p in cur_pts:
                    if p == pt:
                        is_dup = True
                        break
                if not is_dup:
                    cands.add(pt)
                    
    return [list(c) for c in cands]

def _local_search_sa(initial_pts, time_limit):
    start_time = time.time()
    
    cur_pts = initial_pts[:]
    cur_val = _count_ordinary(cur_pts)
    
    best_pts = cur_pts[:]
    best_val = cur_val
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    stall = 0
    max_stall = 100
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 11)
        cands = _generate_candidates(cur_pts, idx)
        
        if not cands:
            continue
            
        # Pick a random candidate or the best one?
        # SA usually picks one. Let's pick the best candidate in the generated set to be more directed,
        # or just a random one to explore more.
        # Given the small number of candidates, let's evaluate all and pick the best, 
        # but use SA acceptance for that best move?
        # Actually, standard SA picks a neighbor. Here we have a set of neighbors.
        # Let's pick a random neighbor from cands.
        
        cand = random.choice(cands)
        new_pts = cur_pts[:]
        new_pts[idx] = cand
        
        if not _valid(new_pts):
            continue
            
        new_val = _count_ordinary(new_pts)
        
        delta = new_val - cur_val
        
        if delta < 0:
            # Accept improvement
            cur_pts = new_pts
            cur_val = new_val
            stall = 0
            if cur_val < best_val:
                best_pts = cur_pts[:]
                best_val = cur_val
        else:
            # Accept with probability exp(-delta/T)
            if T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    cur_pts = new_pts
                    cur_val = new_val
                    stall += 1
            else:
                stall += 1
                
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart temperature if stuck
            if stall > max_stall:
                T = 1.0
                stall = 0
                # Random restart
                for _ in range(3):
                    idx2 = random.randint(0, 11)
                    cur_pts[idx2] = [random.randint(-20, 20), random.randint(-20, 20)]
                    if _valid(cur_pts):
                        cur_val = _count_ordinary(cur_pts)
                        break

    return best_pts, best_val

def _search():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial configurations
    configs = []
    
    # 1. 3x4 Grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 Grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. Hexagon-like
    # Points: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (3,0), (3,1), (3,2)
    configs.append([
        [0,0], [1,0], [2,0],
        [0,1], [1,1], [2,1],
        [0,2], [1,2], [2,2],
        [3,0], [3,1], [3,2]
    ])
    
    # 4. Random
    for _ in range(10):
        configs.append([[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)])
        
    # Filter valid configs
    valid_configs = []
    for c in configs:
        if _valid(c):
            valid_configs.append(c)
            
    if not valid_configs:
        valid_configs = [[[x, y] for x in range(4) for y in range(3)]]
        
    # Evaluate initial configs
    best_init = None
    best_init_val = float('inf')
    for c in valid_configs:
        v = _count_ordinary(c)
        if v < best_init_val:
            best_init_val = v
            best_init = c
            
    if best_init is None:
        best_init = valid_configs[0]
        best_init_val = _count_ordinary(best_init)
        
    # Run SA on best init
    remaining = deadline - time.time()
    if remaining > 5.0:
        pts, val = _local_search_sa(best_init, remaining)
        if val < best_init_val:
            best_init_val = val
            best_init = pts
            
    return best_init

def construct():
    try:
        res = _search()
        if not _valid(res):
            res = [[x, y] for x in range(4) for y in range(3)]
        return res
    except Exception:
        return [[x, y] for x in range(4) for y in range(3)]
