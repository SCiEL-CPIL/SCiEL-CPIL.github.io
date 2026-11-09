"""
Combined approach: Simulated Annealing for global exploration + Local Search for polishing.
Uses fast line counting and validity checks.
"""
import time
import random
import math
from math import gcd

_START = time.time()
_DEADLINE = _START + 115.0
_BOUND = 1000

def _line_key(p, q):
    """Return a canonical key for the line through points p and q."""
    x1, y1 = p
    x2, y2 = q
    dx, dy = x2 - x1, y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    c = x1 * dy - y1 * dx
    return (dx, dy, c)

def _count_lines(points):
    """
    Returns (n4, n5, n3) where:
    n4: number of lines with exactly 4 points
    n5: number of lines with >= 5 points
    n3: number of lines with exactly 3 points
    """
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k not in line_pts:
                line_pts[k] = set()
            line_pts[k].add(i)
            line_pts[k].add(j)
    
    n4 = 0
    n5 = 0
    n3 = 0
    for s in line_pts.values():
        ls = len(s)
        if ls == 4:
            n4 += 1
        elif ls >= 5:
            n5 += 1
        elif ls == 3:
            n3 += 1
    return n4, n5, n3

def _is_valid(points):
    """Check if no 5 points are collinear."""
    n4, n5, n3 = _count_lines(points)
    return n5 == 0

def _score(points):
    """
    Score function for SA.
    Primary: Maximize n4.
    Secondary: Minimize n5 (should be 0 for valid).
    Tertiary: Maximize n3 (more 3-point lines often indicate structure).
    """
    n4, n5, n3 = _count_lines(points)
    # If invalid, score is very low
    if n5 > 0:
        return -10000 - n5
    return n4 * 10 + n3

def _build_ag24():
    """AG(2,4) affine plane over GF(4): 16 points, 16 lines of 4 points each."""
    points = [[x, y] for x in range(4) for y in range(4)]
    return points

def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts

def _seed_random_valid(rng, scale=20):
    points = []
    attempts = 0
    while len(points) < 16 and attempts < 5000:
        attempts += 1
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if [x, y] in points:
            continue
        cand = points + [[x, y]]
        if len(cand) >= 5 and not _is_valid(cand):
            continue
        points.append([x, y])
    while len(points) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if [x, y] not in points:
            points.append([x, y])
    return points[:16]

def _simulated_annealing(points, rng, time_budget):
    """
    Run Simulated Annealing to find a better configuration.
    """
    cur = [list(p) for p in points]
    cur_score = _score(cur)
    best = [list(p) for p in cur]
    best_score = cur_score
    
    T = 50.0
    min_T = 0.1
    cooling = 0.995
    n = len(cur)
    
    start = time.time()
    while time.time() < time_budget:
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Pick a random point to move
        i = rng.randrange(n)
        x0, y0 = cur[i]
        
        # Generate new position
        # Mix of local and global moves
        if rng.random() < 0.7:
            # Local move
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            nx, ny = x0 + dx, y0 + dy
        else:
            # Global move
            nx = rng.randint(-100, 100)
            ny = rng.randint(-100, 100)
            
        if abs(nx) > _BOUND or abs(ny) > _BOUND:
            continue
            
        new_point = [nx, ny]
        if any(new_point == p for j, p in enumerate(cur) if j != i):
            continue
            
        # Apply move
        old_point = cur[i]
        cur[i] = new_point
        
        new_score = _score(cur)
        
        # Acceptance criterion
        if new_score >= cur_score:
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in cur]
        else:
            delta = new_score - cur_score
            # Accept with probability exp(delta / T)
            if rng.random() < math.exp(delta / T):
                cur_score = new_score
            else:
                # Reject
                cur[i] = old_point
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best, best_score

def _local_search_polish(points, rng, time_budget):
    """
    Hill climbing to polish the SA result.
    """
    cur = [list(p) for p in points]
    cur_n4, cur_n5, cur_n3 = _count_lines(cur)
    best = [list(p) for p in cur]
    best_key = (cur_n4, -cur_n5, cur_n3)
    n = len(cur)
    step = 0
    
    while time.time() < time_budget:
        if time.time() > _DEADLINE - 2.0:
            break
        improved = False
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            w = 5
            best_local = None
            best_local_key = None
            
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(p == np_ for p in cur):
                        continue
                    
                    cand = [list(p) for p in cur]
                    cand[i] = np_
                    
                    n4, n5, n3 = _count_lines(cand)
                    if n5 > 0:
                        continue
                    key = (n4, -n5, n3)
                    
                    if best_local_key is None or key > best_local_key:
                        best_local_key = key
                        best_local = np_
            
            if best_local is not None:
                cur_key = (cur_n4, -cur_n5, cur_n3)
                if best_local_key > cur_key:
                    cur[i] = best_local
                    cur_n4, cur_n5, cur_n3 = best_local_key[0], -best_local_key[1], best_local_key[2]
                    improved = True
                    if (cur_n4, -cur_n5, cur_n3) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3)
                        best = [list(p) for p in cur]
        
        step += 1
        if not improved and step > 10:
            # Random kick
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(100):
                nx = x0 + rng.randint(-10, 10)
                ny = y0 + rng.randint(-10, 10)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(p == np_ for p in cur):
                    continue
                cand = [list(p) for p in cur]
                cand[i] = np_
                n4, n5, n3 = _count_lines(cand)
                if n5 > 0:
                    continue
                key = (n4, -n5, n3)
                if key >= (cur_n4, -cur_n5, cur_n3):
                    cur[i] = np_
                    cur_n4, cur_n5, cur_n3 = n4, n5, n3
                    if (cur_n4, -cur_n5, cur_n3) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3)
                        best = [list(p) for p in cur]
                    break
            step = 0
            
    return best, best_key

def construct():
    global _START, _DEADLINE
    rng = random.Random(42)
    
    # Generate diverse seeds
    seeds = [
        _build_ag24(),
        _seed_staggered(),
    ]
    
    for _ in range(5):
        seeds.append(_seed_random_valid(rng, scale=20))
    
    # Filter valid seeds
    valid_seeds = []
    for s in seeds:
        if _is_valid(s):
            valid_seeds.append(s)
    
    if not valid_seeds:
        # Fallback to AG(2,4)
        return _build_ag24()
    
    best_overall_pts = valid_seeds[0]
    best_overall_key = _count_lines(valid_seeds[0])
    
    n_seeds = len(valid_seeds)
    # Allocate time: 70% to SA, 30% to Polish
    total_time = _DEADLINE - time.time()
    sa_time_per_seed = max(2.0, (total_time * 0.7) / n_seeds)
    polish_time_per_seed = max(1.0, (total_time * 0.3) / n_seeds)
    
    for s in valid_seeds:
        if time.time() >= _DEADLINE - 5.0:
            break
            
        # Phase 1: Simulated Annealing
        t_sa_end = min(_DEADLINE - 10.0, time.time() + sa_time_per_seed)
        sa_pts, sa_score = _simulated_annealing(s, rng, t_sa_end)
        
        # Phase 2: Local Search Polish
        t_polish_end = min(_DEADLINE - 2.0, time.time() + polish_time_per_seed)
        final_pts, final_key = _local_search_polish(sa_pts, rng, t_polish_end)
        
        # Update best
        if final_key > best_overall_key:
            best_overall_key = final_key
            best_overall_pts = final_pts
    
    # Final safety check
    if not _is_valid(best_overall_pts):
        return _build_ag24()
        
    for p in best_overall_pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _build_ag24()
            
    return [list(p) for p in best_overall_pts]
