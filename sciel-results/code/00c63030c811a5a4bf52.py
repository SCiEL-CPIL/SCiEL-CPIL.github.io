import time
import random
import math

_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin
_BOUND = 1000

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # c = -A*x - B*y
    # Line: dx*(y-y1) - dy*(x-x1) = 0 => -dy*x + dx*y + (dy*x1 - dx*y1) = 0
    # A = -dy, B = dx, C = dy*x1 - dx*y1
    # Normalize (A, B, C)
    A, B = -dy, dx
    C = dy * x1 - dx * y1
    g2 = _gcd(_gcd(abs(A), abs(B)), abs(C))
    if g2:
        A //= g2
        B //= g2
        C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _build_line_map(points):
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k is None: continue
            if k not in line_map:
                line_map[k] = set()
            line_map[k].add(i)
            line_map[k].add(j)
    return line_map

def _get_score(line_map):
    n4 = 0
    n5_plus = 0
    for s in line_map.values():
        c = len(s)
        if c == 4:
            n4 += 1
        elif c >= 5:
            n5_plus += 1
    return n4, n5_plus

def _is_valid(points):
    if len(set(map(tuple, points))) != len(points):
        return False
    lm = _build_line_map(points)
    for s in lm.values():
        if len(s) >= 5:
            return False
    return True

def _local_search(points, rng, time_limit):
    n = len(points)
    cur = [list(p) for p in points]
    line_map = _build_line_map(cur)
    cur_n4, cur_n5 = _get_score(line_map)
    best_n4, best_n5 = cur_n4, cur_n5
    best_pts = [list(p) for p in cur]
    
    # Objective: Maximize n4, minimize n5 (but n5 must be 0 for validity)
    # Score = n4 - 10*n5
    
    def get_val(n4, n5):
        return n4 - 10 * n5

    cur_val = get_val(cur_n4, cur_n5)
    best_val = cur_val

    # Pre-compute neighbors for speed? No, dynamic.
    
    while time.time() < time_limit:
        improved = False
        # Randomly select a point to move
        i = rng.randrange(n)
        x0, y0 = cur[i]
        
        # Try several random moves for this point
        found_move = False
        for _ in range(20): # 20 attempts per point selection
            # Random offset
            dx = rng.randint(-10, 10)
            dy = rng.randint(-10, 10)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x0 + dx, y0 + dy
            if abs(nx) > _BOUND or abs(ny) > _BOUND:
                continue
            
            new_pt = [nx, ny]
            
            # Check distinctness
            if any(cur[j] == new_pt for j in range(n) if j != i):
                continue
            
            # Save state of affected lines
            # Lines through old_pt
            old_keys = set()
            for j in range(n):
                if j == i: continue
                k = _line_key(cur[i], cur[j])
                if k: old_keys.add(k)
            
            # Lines through new_pt
            new_keys = set()
            for j in range(n):
                if j == i: continue
                k = _line_key(new_pt, cur[j])
                if k: new_keys.add(k)
                
            all_keys = old_keys.union(new_keys)
            
            # Save sizes and sets for these keys
            saved = {}
            for k in all_keys:
                if k in line_map:
                    saved[k] = set(line_map[k])
                else:
                    saved[k] = set()

            # Apply move
            # Remove i from old lines
            for k in old_keys:
                if k in line_map:
                    line_map[k].discard(i)
                    if not line_map[k]:
                        del line_map[k]
            
            cur[i] = new_pt
            
            # Add i to new lines
            for k in new_keys:
                if k not in line_map:
                    line_map[k] = set()
                line_map[k].add(i)
            
            # Evaluate
            new_n4, new_n5 = _get_score(line_map)
            new_val = get_val(new_n4, new_n5)
            
            if new_val > best_val:
                best_val = new_val
                best_n4, best_n5 = new_n4, new_n5
                best_pts = [list(p) for p in cur]
                cur_n4, cur_n5 = new_n4, new_n5
                cur_val = new_val
                improved = True
                found_move = True
                break # Accept and break inner loop to pick new point
            else:
                # Revert
                for k in all_keys:
                    if k in saved:
                        if saved[k]:
                            line_map[k] = saved[k]
                        else:
                            if k in line_map:
                                del line_map[k]
                    else:
                        if k in line_map:
                            del line_map[k]
                cur[i] = [x0, y0]
        
        if not improved and not found_move:
             # Optional: try a larger jump or restart if stuck
             pass
             
    return best_pts, best_n4, best_n5

def _build_seed_grid():
    return [[i, j] for i in range(4) for j in range(4)]

def _build_seed_staggered():
    pts = []
    for j in range(4):
        offset = j % 2
        for i in range(4):
            pts.append([i + offset, j])
    return pts

def _build_seed_random(seed_val):
    rng = random.Random(seed_val)
    pts = set()
    while len(pts) < 16:
        x = rng.randint(-10, 10)
        y = rng.randint(-10, 10)
        pts.add((x, y))
    return [list(p) for p in pts]

def construct():
    global _DEADLINE
    rng = random.Random(42)
    
    seeds = [
        _build_seed_grid(),
        _build_seed_staggered(),
        _build_seed_random(1),
        _build_seed_random(2),
        _build_seed_random(3),
    ]
    
    # Add some perturbed grids
    for _ in range(5):
        p = _build_seed_grid()
        for i in range(16):
            if rng.random() < 0.5:
                p[i][0] += rng.randint(-2, 2)
                p[i][1] += rng.randint(-2, 2)
        if len(set(map(tuple, p))) == 16:
            seeds.append(p)
            
    best_overall_val = -1
    best_overall_pts = None
    
    # Filter valid seeds
    valid_seeds = []
    for s in seeds:
        if _is_valid(s):
            valid_seeds.append(s)
            
    if not valid_seeds:
        return _build_seed_grid()
        
    num_seeds = len(valid_seeds)
    time_per_seed = max(5.0, (_DEADLINE - time.time() - 5.0) / num_seeds)
    
    for s in valid_seeds:
        if time.time() >= _DEADLINE - 5.0:
            break
            
        t_end = min(_DEADLINE - 2.0, time.time() + time_per_seed)
        pts, n4, n5 = _local_search(s, rng, t_end)
        
        if n5 > 0:
            continue # Invalid
            
        val = n4 - 10 * n5
        if val > best_overall_val:
            best_overall_val = val
            best_overall_pts = pts
            
    if best_overall_pts is None:
        return _build_seed_grid()
        
    return best_overall_pts

# Test run
if __name__ == "__main__":
    pts = construct()
    print(f"Time: {time.time() - _START:.2f}s")
    lm = _build_line_map(pts)
    n4, n5 = _get_score(lm)
    print(f"Score: {n4}, Invalid lines: {n5}")
    print(pts)
