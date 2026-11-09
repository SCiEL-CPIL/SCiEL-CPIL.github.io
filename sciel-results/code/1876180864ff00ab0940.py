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

def _deg(pt, s):
    a, b = pt
    return sum(1 for da,db in _NEIGH if (a+da, b+db) in s)

def _swap_delta(cur, cand, s):
    """Return change in edge count when replacing cur with cand in set s (cur in s, cand not in s)."""
    lost = _deg(cur, s)
    a, b = cand
    gained = 0
    for da, db in _NEIGH:
        nb = (a+da, b+db)
        if nb in s and nb != cur:
            gained += 1
    return gained - lost

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _spiral_order(limit):
    """Generate lattice points in spiral order (by hex distance, then angle)."""
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

def _build_hexagonal_patch(n):
    """
    Build a compact hexagonal-like shape with n points.
    """
    points = [(0,0)]
    ring = 1
    while len(points) < n:
        ring_points = []
        for a in range(-ring, ring+1):
            for b in range(-ring, ring+1):
                if _hexdist(a,b) == ring:
                    ring_points.append((a,b))
        
        rp_sorted = []
        for (a,b) in ring_points:
            x = a + 0.5*b
            y = 0.8660254037844386*b
            ang = math.atan2(y, x)
            rp_sorted.append((ang, a, b))
        rp_sorted.sort()
        
        for ang, a, b in rp_sorted:
            if len(points) >= n:
                break
            points.append((a,b))
        
        ring += 1
    
    return points[:n]

def _greedy_construct(n):
    """
    Greedily construct a set of n points by always adding the point
    that maximizes the number of new edges formed.
    """
    pts = [(0,0)]
    s = set(pts)
    
    for _ in range(n-1):
        best_pt = None
        best_gain = -1
        best_dist = 999
        
        cands = set()
        for p in s:
            for da, db in _NEIGH:
                cands.add((p[0]+da, p[1]+db))
        
        cands -= s
        
        for c in cands:
            gain = 0
            for da, db in _NEIGH:
                if (c[0]+da, c[1]+db) in s:
                    gain += 1
            dist = _hexdist(c[0], c[1])
            if gain > best_gain or (gain == best_gain and dist < best_dist):
                best_gain = gain
                best_dist = dist
                best_pt = c
        
        if best_pt is None:
            break
            
        s.add(best_pt)
        pts.append(best_pt)
        
    return pts

def _try_improve(pts, max_time=10.0):
    """Local search to improve edge count via single-point swaps."""
    rng = random.Random(42)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    pool = set()
    for (a,b) in best:
        for r in range(0, 8):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 16:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 10000
    while time.time() - start < max_time and it < 8000000:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            continue
        
        delta = _swap_delta(cur, cand, in_set)
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            stall = 0
            if best_e == 76:
                return best, best_e
        else:
            stall += 1
            if stall > max_stall:
                break
    
    return best, best_e

def _try_improve_2swap(pts, max_time=15.0):
    """Local search with 2-point swaps to escape local optima."""
    rng = random.Random(123)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    pool = set()
    for (a,b) in best:
        for r in range(0, 8):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 16:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 20000
    while time.time() - start < max_time and it < 5000000:
        it += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        cur1 = best[i]
        cur2 = best[j]
        cand1 = pool[rng.randrange(len(pool))]
        cand2 = pool[rng.randrange(len(pool))]
        if cand1 in in_set or cand2 in in_set or cand1 == cand2:
            continue
        if cand1 == cur1 or cand1 == cur2 or cand2 == cur1 or cand2 == cur2:
            continue
        
        s = in_set
        lost = _deg(cur1, s) + _deg(cur2, s)
        if cur2 in s and any((cur2[0]+da, cur2[1]+db) == cur1 for da,db in _NEIGH):
            lost -= 1
        
        s2 = s - {cur1, cur2}
        gained = _deg(cand1, s2) + _deg(cand2, s2)
        a1,b1 = cand1
        a2,b2 = cand2
        for da,db in _NEIGH:
            if (a1+da, b1+db) == cand2:
                gained += 1
                break
        
        delta = gained - lost
        if delta > 0:
            best_e += delta
            best[i] = cand1
            best[j] = cand2
            in_set.discard(cur1)
            in_set.discard(cur2)
            in_set.add(cand1)
            in_set.add(cand2)
            stall = 0
            if best_e == 76:
                return best, best_e
        else:
            stall += 1
            if stall > max_stall:
                break
    
    return best, best_e

def construct():
    deadline = time.time() + 145
    n = 32
    
    # 1. Greedy Construction
    greedy_pts = _greedy_construct(n)
    e_greedy = _count_edges(greedy_pts)
    
    # 2. Hexagonal Patch
    hex_patch = _build_hexagonal_patch(n)
    e_hex = _count_edges(hex_patch)
    
    # 3. Spiral
    order = _spiral_order(12)
    spiral_pts = order[:n]
    e_spiral = _count_edges(spiral_pts)
    
    candidates = [
        (e_greedy, greedy_pts),
        (e_hex, hex_patch),
        (e_spiral, spiral_pts)
    ]
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_e, best = candidates[0]
    
    if best_e == 76:
        return [[a,b] for (a,b) in best]

    # Run local search on the best initial candidate
    remaining = deadline - time.time()
    if remaining > 5.0:
        best1, e1 = _try_improve(best, max_time=min(remaining - 2.0, 20.0))
        if e1 > best_e:
            best, best_e = best1, e1
            if best_e == 76:
                return [[a,b] for (a,b) in best]

    # 2-swap search
    remaining = deadline - time.time()
    if remaining > 5.0:
        best2, e2 = _try_improve_2swap(best, max_time=min(remaining - 2.0, 20.0))
        if e2 > best_e:
            best, best_e = best2, e2
            if best_e == 76:
                return [[a,b] for (a,b) in best]

    # Final polish
    remaining = deadline - time.time()
    if remaining > 2.0:
        best3, e3 = _try_improve(best, max_time=min(remaining - 1.0, 10.0))
        if e3 > best_e:
            best, best_e = best3, e3

    return [[a,b] for (a,b) in best]
