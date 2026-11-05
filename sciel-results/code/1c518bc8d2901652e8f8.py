import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)

def _collinear_check(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)

def _generate_candidate_pool(pts, radius=5):
    """Generate a diverse candidate pool around current points."""
    pool = set()
    # Nearby integer points
    for p in pts:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                pool.add((p[0] + dx, p[1] + dy))
    # Points on lines through pairs (extended)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-4, 5):
                if k == 0:
                    continue
                nx1 = x1 + k * dx
                ny1 = y1 + k * dy
                nx2 = x2 + k * dx
                ny2 = y2 + k * dy
                if abs(nx1) <= 10**9 and abs(ny1) <= 10**9:
                    pool.add((nx1, ny1))
                if abs(nx2) <= 10**9 and abs(ny2) <= 10**9:
                    pool.add((nx2, ny2))
    return list(pool)

def _simulated_annealing(start_pts, pool, max_iters, rng):
    """Simulated annealing to minimize ordinary lines."""
    cur = [list(p) for p in start_pts]
    cur_score = _count_ordinary(cur)
    best = [list(p) for p in cur]
    best_score = cur_score
    T = 3.0  # initial temperature
    T_min = 0.01
    cooling = 0.995  # cooling rate
    
    cur_set = set(tuple(p) for p in cur)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        # Pick a point to move
        idx = rng.randrange(12)
        old_pt = tuple(cur[idx])
        
        # Pick a new position from pool
        cand = tuple(rng.choice(pool))
        if cand == old_pt or cand in cur_set:
            continue
        
        # Compute delta in ordinary lines
        # Remove old point's contribution, add new point's
        # For efficiency, just recompute (n=12 is small)
        new_pts = [list(p) for p in cur]
        new_pts[idx] = list(cand)
        if not _valid(new_pts):
            continue
        new_score = _count_ordinary(new_pts)
        delta = new_score - cur_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and rng.random() < pow(2.718281828, -delta / T)):
            cur = new_pts
            cur_score = new_score
            cur_set = set(tuple(p) for p in cur)
            if cur_score < best_score:
                best = [list(p) for p in cur]
                best_score = cur_score
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def _hill_climb(pts, pool, max_iters, rng):
    """Pure hill climbing (only accept improvements)."""
    cur = [list(p) for p in pts]
    cur_score = _count_ordinary(cur)
    cur_set = set(tuple(p) for p in cur)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        idx = rng.randrange(12)
        old_pt = tuple(cur[idx])
        
        # Try several candidates
        for _ in range(30):
            cand = tuple(rng.choice(pool))
            if cand == old_pt or cand in cur_set:
                continue
            new_pts = [list(p) for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        
        if not improved and it % 500 == 0:
            # Occasionally refresh pool
            if len(pool) < 500:
                pool = _generate_candidate_pool(cur, radius=4)
    
    return cur, cur_score

def _known_configurations():
    """Return several known/diverse starting configurations."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 3x3 grid + 3 extra
    base = [[x, y] for x in range(3) for y in range(3)]
    configs.append(base + [[3, 0], [0, 3], [3, 3]])
    
    # Points on a circle (integer approximations)
    configs.append([[3, 4], [4, 3], [5, 0], [0, 5], [-3, 4], [-4, 3],
                    [-5, 0], [0, -5], [3, -4], [4, -3], [5, -0], [0, -5]])
    # Fix duplicate
    configs[-1] = [[3, 4], [4, 3], [5, 0], [0, 5], [-3, 4], [-4, 3],
                   [-5, 0], [0, -5], [3, -4], [4, -3], [-5, 1], [-1, -5]]
    
    # Random spread
    rng = random.Random(42)
    pts = []
    while len(pts) < 12:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) not in pts:
            pts.append([x, y])
    configs.append(pts)
    
    # Structured: two clusters
    configs.append([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5],[5,6],[6,6],
                    [10,10],[11,10],[10,11],[11,11]])
    
    # Near-collinear with perturbations
    configs.append([[0,0],[1,0],[2,0],[3,1],[4,0],[5,1],[6,0],[7,1],
                    [8,0],[9,1],[10,0],[11,1]])
    
    return configs

def construct():
    rng = random.Random(12345)
    
    # Start with best known configuration
    best_pts = None
    best_score = 999
    
    # Try known configurations
    for cfg in _known_configurations():
        if not _valid(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in cfg]
    
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Generate initial pool
    pool = _generate_candidate_pool(best_pts, radius=5)
    
    # Run simulated annealing from best config
    if time.time() < _DEADLINE - 15:
        max_sa_iters = 200000
        sa_pts, sa_score = _simulated_annealing(best_pts, pool, max_sa_iters, rng)
        if sa_score < best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    # Run hill climbing from best config
    if time.time() < _DEADLINE - 10:
        pool2 = _generate_candidate_pool(best_pts, radius=4)
        hc_pts, hc_score = _hill_climb(best_pts, pool2, 100000, rng)
        if hc_score < best_score:
            best_score = hc_score
            best_pts = hc_pts
    
    # Try SA from other diverse configs
    for cfg in _known_configurations():
        if time.time() > _DEADLINE - 10:
            break
        if not _valid(cfg):
            continue
        cfg_score = _count_ordinary(cfg)
        if cfg_score < best_score + 2:  # Only try if competitive
            pool3 = _generate_candidate_pool(cfg, radius=5)
            sa_pts, sa_score = _simulated_annealing(cfg, pool3, 50000, rng)
            if sa_score < best_score:
                best_score = sa_score
                best_pts = sa_pts
            # Quick hill climb
            if time.time() < _DEADLINE - 8:
                hc_pts, hc_score = _hill_climb(sa_pts, pool3, 30000, rng)
                if hc_score < best_score:
                    best_score = hc_score
                    best_pts = hc_pts
    
    # Final hill climb polish
    if time.time() < _DEADLINE - 5:
        pool4 = _generate_candidate_pool(best_pts, radius=3)
        final_pts, final_score = _hill_climb(best_pts, pool4, 50000, rng)
        if final_score < best_score:
            best_score = final_score
            best_pts = final_pts
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
