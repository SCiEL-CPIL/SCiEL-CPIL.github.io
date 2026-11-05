"""SCiEL-evolved program: line-packing heuristic + exact MIS branch-and-bound scorer."""
import time
import itertools
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = 115.0  # safety margin before the 150s kill

def _norm(v):
    x, y = v
    g = abs(x)
    if g == 0:
        g = abs(y)
    if g == 0:
        return (0, 0)
    if x < 0 or (x == 0 and y < 0):
        g = -g
    return (x // g, y // g)

def _line_key(p, q):
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    n = _norm((dy, -dx))
    c = n[0] * p[0] + n[1] * p[1]
    if c < 0:
        c = -c
    return (n[0], n[1], c)

def _triple_lines(points):
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k not in lines:
                lines[k] = set()
            lines[k].add(i)
            lines[k].add(j)
    return {k: v for k, v in lines.items() if len(v) >= 3}

def _is_valid(points):
    lines = _triple_lines(points)
    for v in lines.values():
        if len(v) >= 4:
            return False
    return True

def _score(points):
    n = len(points)
    lines = _triple_lines(points)
    triples = [frozenset(v) for v in lines.values()]
    
    if not triples:
        return n

    triples_by_pt = [[] for _ in range(n)]
    for t in triples:
        for i in t:
            triples_by_pt[i].append(t)

    best = [0]
    cand = list(range(n))

    def rec(remaining):
        if not remaining:
            return
        if len(remaining) <= best[0]:
            return
        
        remset = set(remaining)
        triple = None
        for i in remaining:
            for t in triples_by_pt[i]:
                if t <= remset:
                    triple = t
                    break
            if triple is not None:
                break
        
        if triple is None:
            if len(remaining) > best[0]:
                best[0] = len(remaining)
            return

        for m in triple:
            new_rem = [x for x in remaining if x != m]
            rec(new_rem)

    rec(cand)
    return best[0]

def _generate_candidate(seed):
    rng = random.Random(seed)
    pts = []
    seen = set()
    
    # 6 rows of 3 points
    for j in range(6):
        # Random x offset for the row
        x0 = rng.randint(-80, 80)
        # Ensure points are within bounds
        # We pick 3 consecutive x's
        if x0 + 2 > 100:
            x0 = 100 - 2
        if x0 < -100:
            x0 = -100
            
        for dx in range(3):
            x = x0 + dx
            y = j * 10  # Spread rows vertically
            if abs(x) > 100 or abs(y) > 100:
                continue
            if (x, y) in seen:
                continue
            pts.append((x, y))
            seen.add((x, y))
            
    # Add 2 random points
    attempts = 0
    while len(pts) < 20 and attempts < 1000:
        attempts += 1
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) in seen:
            continue
        pts.append((x, y))
        seen.add((x, y))
        
    if len(pts) != 20:
        return None
        
    # Check validity
    if not _is_valid(pts):
        return None
        
    return [[p[0], p[1]] for p in pts]

def construct():
    best_pts = None
    best_score = 10**9
    start = time.time()
    
    # Try a range of seeds
    seed = 0
    while time.time() - start < _DEADLINE:
        seed += 1
        cfg = _generate_candidate(seed)
        if cfg is None:
            continue
        
        s = _score(cfg)
        if s < best_score:
            best_score = s
            best_pts = cfg
            if best_score <= 10:
                break
                
    if best_pts is None:
        # Fallback: simple random
        rng = random.Random(42)
        pts = []
        seen = set()
        while len(pts) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) in seen:
                continue
            seen.add((x, y))
            pts.append((x, y))
        if _is_valid(pts):
            best_pts = pts
        else:
            # Absolute fallback: parabola (valid, score 20)
            best_pts = [[x, x * x] for x in range(-10, 10)]
            
    return best_pts
