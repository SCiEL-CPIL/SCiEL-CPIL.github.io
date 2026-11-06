"""Optimized search for 16 integer points maximizing lines with exactly 4 points."""
import random
import math
import time

def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a

def _make_pair_info(pts):
    """Precompute line keys and per-pair other-point sets for 16 points."""
    n = len(pts)
    keys = [[None] * n for _ in range(n)]
    others = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(dx, dy)
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Canonicalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Normal vector (perpendicular)
            nx, ny = -dy, dx
            c = nx * x1 + ny * y1
            key = (dx, dy, nx, ny, c)
            keys[i][j] = key
            keys[j][i] = key
            for k in range(n):
                if k != i and k != j:
                    others[i][j].add(k)
                    others[j][i].add(k)
    return keys, others

def _score(pts, keys, others):
    """Count lines with exactly 4 points; return (count, is_valid)."""
    n = len(pts)
    key_count = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = keys[i][j]
            key_count[k] = key_count.get(k, 0) + 1
    exact4 = 0
    valid = True
    for k, cnt in key_count.items():
        if cnt == 6:  # C(4,2) = 6 pairs for 4 collinear points
            exact4 += 1
        elif cnt > 6:
            # More than 4 points on this line
            valid = False
    # Also check: could have a line with exactly 4 points but also check
    # that no line has exactly 5 points (C(5,2)=10 pairs) or 6 points (C(6,2)=15)
    # cnt > 6 catches all of those since 5 points -> 10 pairs, 6 -> 15, etc.
    # But wait: what about a line with exactly 4 points where some pairs are
    # also on other lines? No, each pair defines exactly one line.
    return exact4, valid

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    # Start with a 4x4 grid
    pts = [[x, y] for x in range(4) for y in range(4)]
    n = len(pts)
    
    # Precompute pair info
    keys, others = _make_pair_info(pts)
    
    # Evaluate initial state
    score, valid = _score(pts, keys, others)
    
    # Local search to fix validity: move points on 5-collinear lines
    # First, let's just run a quick local search to get a valid starting point
    max_local_iters = 5000
    for it in range(max_local_iters):
        if time.time() > deadline:
            break
        if valid and score >= 4:
            break
        # Find a point to move
        if not valid:
            # Find lines with 5+ points
            key_count = {}
            for i in range(n):
                for j in range(i + 1, n):
                    k = keys[i][j]
                    key_count[k] = key_count.get(k, 0) + 1
            bad_keys = [k for k, cnt in key_count.items() if cnt > 6]
            if not bad_keys:
                break
            # Pick a random bad key and move one of its points
            bk = random.choice(bad_keys)
            # Find all pairs with this key
            pair_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if keys[i][j] == bk:
                        pair_list.append((i, j))
            # Pick a random pair and move one endpoint
            pi, pj = random.choice(pair_list)
            move_idx = random.choice([pi, pj])
        else:
            move_idx = random.randint(0, n - 1)
        
        # Try a few random moves
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Move to a nearby point
            ox, oy = pts[move_idx]
            # Random offset in [-3, 3]
            new_x = ox + random.randint(-3, 3)
            new_y = oy + random.randint(-3, 3)
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
            # Check distinctness
            if any(new_x == pts[k][0] and new_y == pts[k][1] for k in range(n) if k != move_idx):
                continue
            
            old = pts[move_idx]
            pts[move_idx] = [new_x, new_y]
            
            # Recompute pair info for this point
            for j in range(n):
                if j == move_idx:
                    continue
                x1, y1 = pts[move_idx]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = _gcd(dx, dy)
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                nx, ny = -dy, dx
                c = nx * x1 + ny * y1
                key = (dx, dy, nx, ny, c)
                keys[move_idx][j] = key
                keys[j][move_idx] = key
                others[move_idx][j] = set()
                others[j][move_idx] = set()
            
            for k in range(n):
                if k == move_idx:
                    continue
                for l in range(n):
                    if l == move_idx or l == k:
                        continue
                    others[move_idx][k].add(l)
                    others[k][move_idx].add(l)
            
            new_score, new_valid = _score(pts, keys, others)
            
            if new_valid and (not valid or new_score > score):
                valid = new_valid
                score = new_score
                break
            else:
                # Revert
                pts[move_idx] = old
                for j in range(n):
                    if j == move_idx:
                        continue
                    x1, y1 = pts[move_idx]
                    x2, y2 = pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    g = _gcd(dx, dy)
                    if g == 0:
                        continue
                    dx //= g
                    dy //= g
                    if dx < 0 or (dx == 0 and dy < 0):
                        dx = -dx
                        dy = -dy
                    nx, ny = -dy, dx
                    c = nx * x1 + ny * y1
                    key = (dx, dy, nx, ny, c)
                    keys[move_idx][j] = key
                    keys[j][move_idx] = key
                    others[move_idx][j] = set()
                    others[j][move_idx] = set()
                for k in range(n):
                    if k == move_idx:
                        continue
                    for l in range(n):
                        if l == move_idx or l == k:
                            continue
                        others[move_idx][k].add(l)
                        others[k][move_idx].add(l)
    
    # Now run simulated annealing from this valid starting point
    if not valid:
        # Fallback: just return the grid (shouldn't happen)
        return [[x, y] for x in range(4) for y in range(4)]
    
    best_score = score
    best_pts = [p[:] for p in pts]
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        ox, oy = pts[idx]
        
        # Generate a new position
        # Sometimes do a small move, sometimes a larger one
        if random.random() < 0.7:
            new_x = ox + random.randint(-4, 4)
            new_y = oy + random.randint(-4, 4)
        else:
            new_x = ox + random.randint(-15, 15)
            new_y = oy + random.randint(-15, 15)
        
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
        
        # Check distinctness
        if any(new_x == pts[k][0] and new_y == pts[k][1] for k in range(n) if k != idx):
            continue
        
        old = pts[idx]
        pts[idx] = [new_x, new_y]
        
        # Recompute pair info for this point
        for j in range(n):
            if j == idx:
                continue
            x1, y1 = pts[idx]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(dx, dy)
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            nx, ny = -dy, dx
            c = nx * x1 + ny * y1
            key = (dx, dy, nx, ny, c)
            keys[idx][j] = key
            keys[j][idx] = key
            others[idx][j] = set()
            others[j][idx] = set()
        
        for k in range(n):
            if k == idx:
                continue
            for l in range(n):
                if l == idx or l == k:
                    continue
                others[idx][k].add(l)
                others[k][idx].add(l)
        
        new_score, new_valid = _score(pts, keys, others)
        
        if new_valid:
            if new_score > score:
                # Accept
                score = new_score
                if score > best_score:
                    best_score = score
                    best_pts = [p[:] for p in pts]
            else:
                # Maybe accept worse
                delta = score - new_score
                if random.random() < math.exp(-delta / max(T, 1e-9)):
                    score = new_score
                else:
                    # Revert
                    pts[idx] = old
                    for j in range(n):
                        if j == idx:
                            continue
                        x1, y1 = pts[idx]
                        x2, y2 = pts[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        g = _gcd(dx, dy)
                        if g == 0:
                            continue
                        dx //= g
                        dy //= g
                        if dx < 0 or (dx == 0 and dy < 0):
                            dx = -dx
                            dy = -dy
                        nx, ny = -dy, dx
                        c = nx * x1 + ny * y1
                        key = (dx, dy, nx, ny, c)
                        keys[idx][j] = key
                        keys[j][idx] = key
                        others[idx][j] = set()
                        others[j][idx] = set()
                    for k in range(n):
                        if k == idx:
                            continue
                        for l in range(n):
                            if l == idx or l == k:
                                continue
                            others[idx][k].add(l)
                            others[k][idx].add(l)
        else:
            # Invalid: revert
            pts[idx] = old
            for j in range(n):
                if j == idx:
                    continue
                x1, y1 = pts[idx]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = _gcd(dx, dy)
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                nx, ny = -dy, dx
                c = nx * x1 + ny * y1
                key = (dx, dy, nx, ny, c)
                keys[idx][j] = key
                keys[j][idx] = key
                others[idx][j] = set()
                others[j][idx] = set()
            for k in range(n):
                if k == idx:
                    continue
                for l in range(n):
                    if l == idx or l == k:
                        continue
                    others[idx][k].add(l)
                    others[k][idx].add(l)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return best found
    return best_pts
