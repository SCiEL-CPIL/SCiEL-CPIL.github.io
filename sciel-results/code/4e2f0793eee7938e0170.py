import time
import random
import numpy as np
from collections import defaultdict

_START = time.time()
_DEADLINE = 120.0

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    a = y1 - y2
    b = x2 - x1
    c = -(a * x1 + b * y1)
    g = _gcd(_gcd(a, b), c)
    if g:
        a //= g; b //= g; c //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a; b = -b; c = -c
    return (a, b, c)

def _count4(pts):
    n = len(pts)
    line_pts = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            line_pts[k].add(i)
            line_pts[k].add(j)
    cnt4 = 0
    has5 = False
    for k, s in line_pts.items():
        m = len(s)
        if m == 4:
            cnt4 += 1
        elif m >= 5:
            has5 = True
    return cnt4, has5

def _valid(pts):
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

def _line_neighbor(pts, rng):
    """Move a point to lie on a line defined by two other points."""
    new = [list(p) for p in pts]
    idx = rng.randrange(16)
    others = [i for i in range(16) if i != idx]
    i1, i2 = rng.sample(others, 2)
    p1, p2 = pts[i1], pts[i2]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    step_x = dx // g
    step_y = dy // g
    base_x, base_y = p1[0], p1[1]
    for t in range(-30, 31):
        if t == 0:
            continue
        nx = base_x + t * step_x
        ny = base_y + t * step_y
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        cand = [list(p) for p in pts]
        cand[idx] = [nx, ny]
        if len(set(map(tuple, cand))) != 16:
            continue
        return cand
    return None

def _three_point_line_neighbor(pts, rng):
    """Move a point to complete a line with exactly 3 existing points to make 4."""
    n = 16
    idx = rng.randrange(n)
    others = [i for i in range(n) if i != idx]
    line_counts = defaultdict(set)
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            oi, oj = others[i], others[j]
            k = _line_key(pts[oi], pts[oj])
            line_counts[k].add(oi)
            line_counts[k].add(oj)
    
    candidate_lines = []
    for k, s in line_counts.items():
        if len(s) == 3:
            candidate_lines.append(k)
    
    if not candidate_lines:
        return None
    
    k = rng.choice(candidate_lines)
    a, b, c = k
    existing = set(tuple(p) for p in pts)
    
    if a == 0:
        if b == 0:
            return None
        if c % b != 0:
            return None
        y_val = c // b
        for x_val in range(-1000, 1001):
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    elif b == 0:
        if a == 0:
            return None
        if c % a != 0:
            return None
        x_val = c // a
        for y_val in range(-1000, 1001):
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    else:
        def _ext_gcd(aa, bb):
            if bb == 0:
                return (1, 0, aa)
            x1, y1, g = _ext_gcd(bb, aa % bb)
            return (y1, x1 - (aa // bb) * y1, g)
        
        g = _gcd(a, b)
        if c % g != 0:
            return None
        a_div, b_div, c_div = a // g, b // g, c // g
        x0, y0, _ = _ext_gcd(a_div, b_div)
        x0 *= c_div
        y0 *= c_div
        
        for t in range(-100, 101):
            x_val = x0 + b_div * t
            y_val = y0 - a_div * t
            if abs(x_val) > 1000 or abs(y_val) > 1000:
                continue
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    return None

def _two_point_extension(pts, rng):
    """Find lines with exactly 2 points and extend them by moving a point to create a 3-point line."""
    n = 16
    idx = rng.randrange(n)
    others = [i for i in range(n) if i != idx]
    line_counts = defaultdict(set)
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            oi, oj = others[i], others[j]
            k = _line_key(pts[oi], pts[oj])
            line_counts[k].add(oi)
            line_counts[k].add(oj)
    
    candidate_lines = [k for k, s in line_counts.items() if len(s) == 2]
    if not candidate_lines:
        return None
    
    k = rng.choice(candidate_lines)
    a, b, c = k
    existing = set(tuple(p) for p in pts)
    
    if a == 0:
        if b == 0:
            return None
        if c % b != 0:
            return None
        y_val = c // b
        for x_val in range(-1000, 1001):
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    elif b == 0:
        if a == 0:
            return None
        if c % a != 0:
            return None
        x_val = c // a
        for y_val in range(-1000, 1001):
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    else:
        def _ext_gcd(aa, bb):
            if bb == 0:
                return (1, 0, aa)
            x1, y1, g = _ext_gcd(bb, aa % bb)
            return (y1, x1 - (aa // bb) * y1, g)
        
        g = _gcd(a, b)
        if c % g != 0:
            return None
        a_div, b_div, c_div = a // g, b // g, c // g
        x0, y0, _ = _ext_gcd(a_div, b_div)
        x0 *= c_div
        y0 *= c_div
        
        for t in range(-100, 101):
            x_val = x0 + b_div * t
            y_val = y0 - a_div * t
            if abs(x_val) > 1000 or abs(y_val) > 1000:
                continue
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    return None

def _neighbor(pts, rng):
    """Move one point to a nearby random integer location."""
    new = [list(p) for p in pts]
    idx = rng.randrange(16)
    base = new[idx]
    for _ in range(40):
        dx = rng.randint(-20, 20)
        dy = rng.randint(-20, 20)
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

def _pairwise_swap(pts, rng):
    """Swap coordinates of two points to explore the search space."""
    cand = [list(p) for p in pts]
    i, j = rng.sample(range(16), 2)
    cand[i], cand[j] = cand[j], cand[i]
    if len(set(map(tuple, cand))) != 16:
        return None
    return cand

def _local_search_v2(start, rng, max_iters, deadline):
    """Enhanced local search with more diverse moves."""
    cur = [list(p) for p in start]
    cur_s = _score(cur)
    if cur_s < 0:
        cur = _grid44()
        cur_s = _score(cur)
    best = [list(p) for p in cur]
    best_s = cur_s
    T = 1.0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        r = rng.random()
        if r < 0.20:
            nb = _neighbor(cur, rng)
        elif r < 0.35:
            nb = _line_neighbor(cur, rng)
        elif r < 0.55:
            nb = _three_point_line_neighbor(cur, rng)
        elif r < 0.75:
            nb = _two_point_extension(cur, rng)
        elif r < 0.85:
            nb = _pairwise_swap(cur, rng)
        else:
            nb = [list(p) for p in cur]
            k = rng.randint(2, 3)
            for _ in range(k):
                idx = rng.randrange(16)
                x = rng.randint(-100, 100)
                y = rng.randint(-100, 100)
                nb[idx] = [x, y]
            if len(set(map(tuple, nb))) != 16:
                continue
        
        if nb is None:
            continue
        nb_s = _score(nb)
        if nb_s < 0:
            continue
        if nb_s > cur_s:
            cur = nb
            cur_s = nb_s
            if cur_s > best_s:
                best = [list(p) for p in cur]
                best_s = cur_s
        else:
            d = nb_s - cur_s
            if d < 0 and rng.random() < np.exp(d / max(T, 1e-6)):
                cur = nb
                cur_s = nb_s
        T *= 0.9995
    return best, best_s

def _random_start():
    pts = []
    seen = set()
    while len(pts) < 16:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        pts.append([x, y])
    return pts

def construct():
    deadline = _START + _DEADLINE
    rng = random.Random(42)
    np.random.seed(42)

    best = _grid44()
    best_s = _score(best)  # 10 for the 4x4 grid
    
    # Phase 1: local search from the grid
    if time.time() < deadline - 10:
        b, s = _local_search_v2(best, rng, 40000, deadline - 5)
        if s > best_s:
            best, best_s = b, s

    # Phase 2: multiple random starts
    attempts = 0
    while time.time() < deadline - 8:
        attempts += 1
        start = _random_start()
        b, s = _local_search_v2(start, rng, 3000, deadline - 5)
        if s > best_s:
            best, best_s = b, s
        if attempts % 4 == 0 and time.time() < deadline - 5:
            b2, s2 = _local_search_v2(best, rng, 3000, deadline - 5)
            if s2 > best_s:
                best, best_s = b2, s2

    # Final safety: ensure validity
    if not _valid(best):
        best = _grid44()
    return best
