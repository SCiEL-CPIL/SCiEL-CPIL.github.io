import time
import random
import numpy as np
from collections import defaultdict

_START = time.time()
_DEADLINE = 120.0   # evaluator kills at 150s; keep margin

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p, q):
    """Canonical integer line (a,b,c) for a*x + b*y = c through p,q, reduced."""
    x1, y1 = p
    x2, y2 = q
    a = y1 - y2
    b = x2 - x1
    c = -(a * x1 + b * y1)
    g = _gcd(_gcd(a, b), c)
    if g:
        a //= g; b //= g; c //= g
    # sign canonical: first non-zero of (a,b) positive
    if a < 0 or (a == 0 and b < 0):
        a = -a; b = -b; c = -c
    return (a, b, c)

def _count4(pts):
    """Return (num_lines_exactly4, has_five_collinear) for list of 16 points."""
    n = len(pts)
    line_set = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            line_set[k].add(i)
            line_set[k].add(j)
    cnt4 = 0
    has5 = False
    for k, s in line_set.items():
        m = len(s)
        if m == 4:
            cnt4 += 1
        elif m >= 5:
            has5 = True
    return cnt4, has5

def _valid(pts):
    """Distinct, in range, no five collinear."""
    if len(set(map(tuple, pts))) != 16:
        return False
    for (x, y) in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, has5 = _count4(pts)
    return not has5

def _score(pts):
    if not _valid(pts):
        return -1
    c, _ = _count4(pts)
    return c

def _grid44():
    return [[x, y] for x in range(4) for y in range(4)]

def _random_start():
    pts = []
    seen = set()
    while len(pts) < 16:
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        pts.append([x, y])
    return pts

def _neighbor(pts, rng, step_size=6):
    """Return a new point list with one point moved. Uses smart moves to create 4-point lines."""
    n = len(pts)
    idx = rng.randrange(n)
    
    # Strategy 1: Move a point to complete a 4-point line
    # Find pairs of existing points and check if moving idx to a point on their line helps
    if rng.random() < 0.4:
        # Pick two other points
        others = [i for i in range(n) if i != idx]
        i1 = rng.choice(others)
        i2 = rng.choice(others)
        if i1 == i2:
            return None
        
        p1, p2 = pts[i1], pts[i2]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Try points on the line through p1, p2 at various offsets
        # Line: (x1 + t*dx, y1 + t*dy) for integer t
        # We need to find integer t such that the point is valid and distinct
        g = _gcd(abs(dx), abs(dy))
        if g == 0:
            return None
        step_x = dx // g
        step_y = dy // g
        
        # Try several t values
        for t in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            nx = p1[0] + t * step_x
            ny = p1[1] + t * step_y
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            # Check distinctness
            if any(nx == p[0] and ny == p[1] for p in pts):
                continue
            cand = [list(p) for p in pts]
            cand[idx] = [nx, ny]
            return cand
        return None
    
    # Strategy 2: Move a point to be collinear with two others
    if rng.random() < 0.3:
        others = [i for i in range(n) if i != idx]
        i1 = rng.choice(others)
        i2 = rng.choice(others)
        if i1 == i2:
            return None
        
        p1, p2 = pts[i1], pts[i2]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Find a point on this line that's close to pts[idx]
        # Parametric: (x1 + t*dx, y1 + t*dy)
        # Minimize distance to pts[idx]
        base = pts[idx]
        # Project base onto line
        if dx == 0 and dy == 0:
            return None
        # t = ((base - p1) . (p2 - p1)) / |p2-p1|^2
        dot = (base[0] - p1[0]) * dx + (base[1] - p1[1]) * dy
        sq = dx * dx + dy * dy
        t = dot / sq
        
        # Try integer t values near the projection
        t_int = int(round(t))
        for dt in range(-2, 3):
            tt = t_int + dt
            nx = p1[0] + tt * dx
            ny = p1[1] + tt * dy
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            if any(nx == p[0] and ny == p[1] for p in pts):
                continue
            cand = [list(p) for p in pts]
            cand[idx] = [nx, ny]
            return cand
        return None
    
    # Strategy 3: Random nearby move
    base = pts[idx]
    for _ in range(20):
        dx = rng.randint(-step_size, step_size)
        dy = rng.randint(-step_size, step_size)
        if dx == 0 and dy == 0:
            continue
        nx = base[0] + dx
        ny = base[1] + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        cand = [list(p) for p in pts]
        cand[idx] = [nx, ny]
        if len(set(map(tuple, cand))) != 16:
            continue
        return cand
    return None

def _local_search(start, rng, max_iters, deadline, T0=1.0, cooling=0.999):
    cur = [list(p) for p in start]
    cur_s = _score(cur)
    if cur_s < 0:
        cur = _grid44()
        cur_s = _score(cur)
    best = [list(p) for p in cur]
    best_s = cur_s
    T = T0
    stall_count = 0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        nb = _neighbor(cur, rng)
        if nb is None:
            continue
        nb_s = _score(nb)
        if nb_s < 0:
            # invalid (five collinear or dup) -> reject
            stall_count += 1
            continue
        if nb_s > cur_s:
            cur = nb
            cur_s = nb_s
            stall_count = 0
            if cur_s > best_s:
                best = [list(p) for p in cur]
                best_s = cur_s
        else:
            # simulated annealing acceptance
            d = nb_s - cur_s
            if d < 0 and rng.random() < np.exp(d / max(T, 1e-6)):
                cur = nb
                cur_s = nb_s
            stall_count += 1
        T *= cooling
        
        # If stalled for too long, perturb to escape local optimum
        if stall_count > 50:
            cur = _perturb(cur, rng, 1)
            cur_s = _score(cur)
            if cur_s < 0:
                cur = [list(p) for p in best]
                cur_s = best_s
            stall_count = 0
    return best, best_s

def _perturb(pts, rng, num_moves=2):
    """Apply num_moves random perturbations to pts."""
    new = [list(p) for p in pts]
    for _ in range(num_moves):
        idx = rng.randrange(16)
        dx = rng.randint(-15, 15)
        dy = rng.randint(-15, 15)
        nx = new[idx][0] + dx
        ny = new[idx][1] + dy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            new[idx] = [nx, ny]
    return new

def construct():
    deadline = _START + _DEADLINE
    rng = random.Random(12345)
    np.random.seed(12345)

    best = _grid44()
    best_s = _score(best)  # 10

    # Phase 1: local search from the grid
    if time.time() < deadline - 10:
        b, s = _local_search(_grid44(), rng, 20000, deadline - 10)
        if s > best_s:
            best, best_s = b, s

    # Phase 2: multiple random starts with occasional perturbation of best
    attempts = 0
    while time.time() < deadline - 15:
        attempts += 1
        start = _random_start()
        b, s = _local_search(start, rng, 4000, deadline - 15)
        if s > best_s:
            best, best_s = b, s
        # occasionally re-seed from current best to refine
        if attempts % 5 == 0 and time.time() < deadline - 15:
            perturbed = _perturb(best, rng, 2)
            b2, s2 = _local_search(perturbed, rng, 3000, deadline - 15)
            if s2 > best_s:
                best, best_s = b2, s2

    # Phase 3: Final polish - intensive local search from best
    if time.time() < deadline - 2:
        b, s = _local_search(best, rng, 10000, deadline - 2)
        if s > best_s:
            best, best_s = b, s

    # Final safety: ensure validity
    if not _valid(best):
        best = _grid44()
    return best
