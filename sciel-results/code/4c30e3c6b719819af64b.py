import time
import math

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
    # Edges lost: neighbors of cur in s (excluding cur itself, which isn't a neighbor)
    lost = _deg(cur, s)
    # Edges gained: neighbors of cand in s, excluding cur (cur is being removed)
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
    For triangular lattice, the optimal packing for maximizing edges is a hexagonal number pattern.
    We build concentric hexagonal rings and fill them to reach exactly n points.
    """
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # H_k = 3*k*(k-1) + 1 for k >= 1
    # We want to build the shape by adding rings.
    
    # Start with center
    points = [(0,0)]
    
    # Add rings one by one
    ring = 1
    while len(points) < n:
        # The ring at distance 'ring' has 6*ring points
        ring_points = []
        # Walk around the hexagon at this distance
        # Starting from (ring, 0) and going around
        # The 6 directions for moving to next ring position
        # Actually, let's use the spiral order approach for the ring
        
        # For a hexagonal ring at distance r, the points are:
        # (r, -r) to (r, 0): direction (1,0) -> wait, let me think more carefully
        # In axial coordinates (a,b) with cube constraint a+b+c=0 where c=-a-b:
        # A hexagonal ring at distance r consists of points where max(|a|,|b|,|c|)=r
        
        for a in range(-ring, ring+1):
            for b in range(-ring, ring+1):
                if _hexdist(a,b) == ring:
                    ring_points.append((a,b))
        
        # Sort ring points by angle to get consistent ordering
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

def _try_improve(pts, max_time=8.0):
    """Local search to improve edge count via single-point swaps with incremental delta."""
    import random
    rng = random.Random(42)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build candidate pool
    pool = set()
    for (a,b) in best:
        for r in range(0, 6):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 12:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 8000
    while time.time() - start < max_time and it < 5000000:
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
        else:
            stall += 1
            if stall > max_stall:
                break
    
    return best, best_e

def _try_improve_2swap(pts, max_time=5.0):
    """Local search with 2-point swaps to escape local optima."""
    import random
    rng = random.Random(123)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build candidate pool
    pool = set()
    for (a,b) in best:
        for r in range(0, 6):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 12:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 15000
    while time.time() - start < max_time and it < 3000000:
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
        
        # Compute delta for simultaneous removal of cur1, cur2 and addition of cand1, cand2
        s = in_set
        # Edges lost: edges incident to cur1 or cur2 (within s), minus double-counted edge between them if exists
        lost = _deg(cur1, s) + _deg(cur2, s)
        if cur2 in s and any((cur2[0]+da, cur2[1]+db) == cur1 for da,db in _NEIGH):
            lost -= 1  # edge between cur1 and cur2 counted twice
        
        # Edges gained: edges from cand1/cand2 to remaining set (s minus cur1,cur2), plus edge between cand1,cand2
        s2 = s - {cur1, cur2}
        gained = _deg(cand1, s2) + _deg(cand2, s2)
        a1,b1 = cand1
        a2,b2 = cand2
        # Check if cand1 and cand2 are adjacent
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
        else:
            stall += 1
            if stall > max_stall:
                break
    
    return best, best_e

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n = 32
    
    # Strategy 1: Build hexagonal patch directly
    hex_patch = _build_hexagonal_patch(n)
    e_hex = _count_edges(hex_patch)
    
    # Strategy 2: Spiral fill
    order = _spiral_order(12)
    spiral_pts = order[:n]
    e_spiral = _count_edges(spiral_pts)
    
    # Pick the better starting point
    if e_hex >= e_spiral:
        best = hex_patch
        best_e = e_hex
    else:
        best = spiral_pts
        best_e = e_spiral
    
    # Local search from both starting points
    best1, e1 = _try_improve(hex_patch, max_time=5.0)
    best2, e2 = _try_improve(spiral_pts, max_time=5.0)
    
    if e1 > best_e:
        best, best_e = best1, e1
    if e2 > best_e:
        best, best_e = best2, e2
    
    # 2-swap search to escape local optima
    remaining = deadline - time.time()
    if remaining > 3.0:
        best4, e4 = _try_improve_2swap(best, max_time=min(remaining - 2.0, 10.0))
        if e4 > best_e:
            best, best_e = best4, e4
    
    # Additional local search from the current best with more time
    remaining = deadline - time.time()
    if remaining > 2.0:
        best3, e3 = _try_improve(best, max_time=min(remaining - 1.0, 15.0))
        if e3 > best_e:
            best, best_e = best3, e3
    
    # Final 2-swap polish
    remaining = deadline - time.time()
    if remaining > 2.0:
        best5, e5 = _try_improve_2swap(best, max_time=min(remaining - 1.0, 8.0))
        if e5 > best_e:
            best, best_e = best5, e5
    
    return [[a,b] for (a,b) in best]
