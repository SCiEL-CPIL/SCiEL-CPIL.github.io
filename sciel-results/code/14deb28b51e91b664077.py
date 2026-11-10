"""Rewritten program: search for 16 integer points maximizing 4-point lines."""
import time
import random
import math
from itertools import combinations

# Deadline guard
_START = time.time()
_DEADLINE = _START + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _normalize_line(p1, p2):
    """Return normalized line representation (a, b, c) for ax+by+c=0."""
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = -(a * p1[0] + b * p1[1])
    # Normalize sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    # Reduce by gcd
    from math import gcd
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    return (a, b, c)

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points. Returns (count, max_collinear)."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _normalize_line(points[i], points[j])
            if key not in lines:
                lines[key] = set([i, j])
            else:
                lines[key].add(i)
                lines[key].add(j)
    four_count = 0
    max_collinear = 0
    for key, pts in lines.items():
        cnt = len(pts)
        if cnt > max_collinear:
            max_collinear = cnt
        if cnt == 4:
            four_count += 1
    return four_count, max_collinear

def _score(points):
    """Score: number of 4-point lines. Invalid if 5+ collinear."""
    four_count, max_collinear = _count_four_lines(points)
    if max_collinear >= 5:
        return -1000  # penalty
    return four_count

def _random_point():
    """Generate a random integer point with |x|,|y| <= 1000."""
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _init_grid():
    """Initialize with a 4x4 grid (known good: 10 four-point lines)."""
    return [[x, y] for x in range(4) for y in range(4)]

def _init_grid_offset():
    """4x4 grid with offset."""
    ox, oy = random.randint(-500, 500), random.randint(-500, 500)
    return [[x + ox, y + oy] for x in range(4) for y in range(4)]

def _init_random():
    """16 random distinct points."""
    pts = []
    seen = set()
    while len(pts) < 16:
        p = _random_point()
        if tuple(p) not in seen:
            seen.add(tuple(p))
            pts.append(p)
    return pts

def _perturb(points):
    """Move one point to a new random location."""
    new_pts = [p[:] for p in points]
    idx = random.randint(0, 15)
    while True:
        p = _random_point()
        if all(tuple(p) != tuple(q) for q in new_pts):
            break
    new_pts[idx] = p
    return new_pts

def _perturb_small(points):
    """Move one point by a small amount."""
    new_pts = [p[:] for p in points]
    idx = random.randint(0, 15)
    dx = random.randint(-10, 10)
    dy = random.randint(-10, 10)
    np_ = [new_pts[idx][0] + dx, new_pts[idx][1] + dy]
    if abs(np_[0]) <= 1000 and abs(np_[1]) <= 1000 and all(tuple(np_) != tuple(q) for q in new_pts if q is not new_pts[idx]):
        new_pts[idx] = np_
    return new_pts

def _simulated_annealing(points, max_iters):
    """Simulated annealing to improve the configuration."""
    current = [p[:] for p in points]
    current_score = _score(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        
        new_pts = _perturb(current)
        new_score = _score(new_pts)
        
        if new_score >= current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [p[:] for p in current]
                best_score = current_score
        else:
            # Accept with probability exp((new - cur)/T)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current = new_pts
                current_score = new_score
        
        T = max(T_min, T * cooling)
    
    return best, best_score

def _local_search(points, max_iters):
    """Hill climbing: only accept improvements."""
    current = [p[:] for p in points]
    current_score = _score(current)
    best = [p[:] for p in current]
    best_score = current_score
    no_improve = 0
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        
        improved = False
        for idx in range(16):
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in current]
                    np_ = [current[idx][0] + dx, current[idx][1] + dy]
                    if abs(np_[0]) > 1000 or abs(np_[1]) > 1000:
                        continue
                    if any(tuple(np_) == tuple(q) for j, q in enumerate(current) if j != idx):
                        continue
                    new_pts[idx] = np_
                    s = _score(new_pts)
                    if s > current_score:
                        current = new_pts
                        current_score = s
                        improved = True
                        if s > best_score:
                            best = [p[:] for p in current]
                            best_score = s
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 10:
                break
    
    return best, best_score

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    
    random.seed(42)
    
    best_pts = None
    best_score = -1
    
    # Try multiple initializations and optimization strategies
    initializations = [
        _init_grid(),
        _init_grid_offset(),
        _init_grid_offset(),
        _init_random(),
        _init_random(),
    ]
    
    for init_pts in initializations:
        if _time_left() < 10:
            break
        
        # First, do local search
        ls_pts, ls_score = _local_search(init_pts, max_iters=50)
        if ls_score > best_score:
            best_score = ls_score
            best_pts = [p[:] for p in ls_pts]
        
        # Then, simulated annealing from the local search result
        sa_pts, sa_score = _simulated_annealing(ls_pts, max_iters=2000)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = [p[:] for p in sa_pts]
        
        # Another local search pass
        if _time_left() > 15:
            ls2_pts, ls2_score = _local_search(sa_pts, max_iters=30)
            if ls2_score > best_score:
                best_score = ls2_score
                best_pts = [p[:] for p in ls2_pts]
    
    # Additional random restarts if time permits
    while _time_left() > 20:
        init_pts = _init_random()
        sa_pts, sa_score = _simulated_annealing(init_pts, max_iters=1000)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = [p[:] for p in sa_pts]
    
    # Final polish
    if best_pts is None:
        best_pts = _init_grid()
    
    if _time_left() > 5:
        final_pts, final_score = _local_search(best_pts, max_iters=20)
        if final_score > best_score:
            best_pts = [p[:] for p in final_pts]
            best_score = final_score
    
    # Validate
    four_count, max_collinear = _count_four_lines(best_pts)
    if max_collinear >= 5:
        # Fallback to 4x4 grid
        best_pts = _init_grid()
    
    return best_pts
