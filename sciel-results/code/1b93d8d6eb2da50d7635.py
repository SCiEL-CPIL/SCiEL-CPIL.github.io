import time
import math
import random

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _spiral_order(limit):
    R = limit
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            hd = _hexdist(a,b)
            x = a + 0.5*b
            y = 0.8660254037844386*b
            ang = math.atan2(y, x)
            cands.append((hd, ang, a, b))
    cands.sort()
    out = []
    for hd, ang, a, b in cands:
        out.append((a,b))
    return out

def _known_optimal():
    """Return a known 32-point configuration with 76 edges.
    
    Use a hexagon of radius 3 (37 points) and remove 5 boundary points
    that are corners (degree 2) to minimize edge loss.
    """
    pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if _hexdist(a,b) <= 3:
                pts.append((a,b))
    s = set(pts)
    deg = {}
    for p in pts:
        d = 0
        for da,db in _NEIGH:
            if (p[0]+da, p[1]+db) in s:
                d += 1
        deg[p] = d
    # Remove 5 points with lowest degree (corners have degree 2)
    pts_sorted = sorted(pts, key=lambda p: (deg[p], _hexdist(p[0],p[1])))
    to_remove = set(pts_sorted[:5])
    result = [p for p in pts if p not in to_remove]
    e = _count_edges(result)
    if e >= 76:
        return result
    return None

def _greedy_construct(n=32):
    """Greedy construction: add points one by one, each maximizing new edges."""
    order = _spiral_order(12)
    selected = [order[0]]
    s = set(selected)
    
    for _ in range(1, n):
        best_pt = None
        best_edges = -1
        for pt in order:
            if pt in s:
                continue
            edges = 0
            for da,db in _NEIGH:
                if (pt[0]+da, pt[1]+db) in s:
                    edges += 1
            if edges > best_edges:
                best_edges = edges
                best_pt = pt
        if best_pt is None:
            break
        selected.append(best_pt)
        s.add(best_pt)
    
    return selected

def _local_search(pts, n=32, deadline=None):
    """Local search with incremental edge counting and larger candidate pool."""
    if deadline is None:
        deadline = time.time() + 115
    
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build a larger candidate pool
    pool = set()
    for (a,b) in best:
        for da in range(-6, 7):
            for db in range(-6, 7):
                na, nb = a+da, b+db
                if abs(na) <= 64 and abs(nb) <= 64 and _hexdist(na, nb) <= 10:
                    pool.add((na, nb))
    pool = list(pool)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            continue
        
        # Count edges lost by removing cur
        edges_lost = 0
        for da, db in _NEIGH:
            if (cur[0]+da, cur[1]+db) in in_set:
                edges_lost += 1
        
        # Count edges gained by adding cand
        edges_gained = 0
        for da, db in _NEIGH:
            nb = (cand[0]+da, cand[1]+db)
            if nb in in_set and nb != cur:
                edges_gained += 1
        
        delta = edges_gained - edges_lost
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            if best_e >= 76:
                break
    
    return best, best_e

def construct():
    n = 32
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Try known optimal first
    opt = _known_optimal()
    if opt is not None:
        e = _count_edges(opt)
        if e >= 76:
            return [[a,b] for (a,b) in opt]
    
    # Try greedy construction
    greedy = _greedy_construct(n)
    e_greedy = _count_edges(greedy)
    if e_greedy >= 76:
        return [[a,b] for (a,b) in greedy]
    
    # Try spiral order
    order = _spiral_order(10)
    base = order[:n]
    e_base = _count_edges(base)
    if e_base >= 76:
        return [[a,b] for (a,b) in base]
    
    # Try spiral with skips
    for skip in range(1, 6):
        variant = order[skip:skip+n]
        e2 = _count_edges(variant)
        if e2 >= 76:
            return [[a,b] for (a,b) in variant]
    
    # Local search from best starting point
    best_start = None
    best_start_e = 0
    for candidate in [base, greedy]:
        e_c = _count_edges(candidate)
        if e_c > best_start_e:
            best_start_e = e_c
            best_start = candidate
    
    result, final_e = _local_search(best_start, n, deadline)
    
    # Verify and return
    if len(set(result)) != n:
        # Fallback to base
        return [[a,b] for (a,b) in base]
    
    return [[a,b] for (a,b) in result]
