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

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _spiral_order(limit):
    """Generate lattice points in spiral order."""
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

def _greedy_construct(n, order, start_pt=(0,0)):
    """
    Greedily construct a set of n points starting from start_pt.
    At each step, add the point from 'order' that maximizes the number of edges to the current set.
    """
    s = {start_pt}
    pts = [start_pt]
    # Pre-compute neighbors for all points in order? No, just check on the fly.
    # To speed up, we can filter 'order' to only points within a reasonable distance.
    # But for n=32, checking all is fine if order is small.
    
    # We need to find the best point to add.
    # We can iterate through 'order' and pick the one with max degree.
    # However, as the set grows, the best point might be further away?
    # No, compactness is key.
    
    # Optimization: Only consider points that are adjacent to the current set or close to it.
    # Let's just scan the whole list each time, but break early if we find a point with max possible degree (6)?
    # No, max degree is 6, but we want to maximize edges.
    
    for _ in range(n - 1):
        best_pt = None
        best_deg = -1
        # We can restrict search to points with hexdist <= current_max_hexdist + 1
        # But for simplicity and safety, scan all candidates not in s.
        # To make it faster, we can maintain a list of "active" candidates.
        
        # Simple approach: scan 'order'
        for p in order:
            if p in s:
                continue
            d = _deg(p, s)
            if d > best_deg:
                best_deg = d
                best_pt = p
                if best_deg == 6:
                    break # Max possible degree found
        
        if best_pt is None:
            # Should not happen if order is large enough
            break
            
        s.add(best_pt)
        pts.append(best_pt)
        
    return pts

def _deterministic_local_search(pts, max_iter=100):
    """
    Deterministic local search: try all single-point swaps that improve the score.
    Repeat until no improvement is found.
    """
    n = len(pts)
    best = list(pts)
    in_set = set(best)
    best_e = _count_edges(best)
    
    # Candidate pool: points near the current set
    pool = set()
    for (a,b) in best:
        for da in range(-2, 3):
            for db in range(-2, 3):
                pool.add((a+da, b+db))
    pool.difference_update(in_set)
    pool = list(pool)
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n):
            cur = best[i]
            best_delta = 0
            best_cand = None
            for cand in pool:
                if cand in in_set:
                    continue
                # Calculate delta
                lost = _deg(cur, in_set)
                gained = 0
                a, b = cand
                for da, db in _NEIGH:
                    nb = (a+da, b+db)
                    if nb in in_set and nb != cur:
                        gained += 1
                delta = gained - lost
                if delta > best_delta:
                    best_delta = delta
                    best_cand = cand
            
            if best_delta > 0 and best_cand is not None:
                # Apply swap
                best_e += best_delta
                best[i] = best_cand
                in_set.discard(cur)
                in_set.add(best_cand)
                # Update pool: add neighbors of new point, remove cur if not neighbor of others?
                # For simplicity, rebuild pool or just add new neighbors.
                # Adding neighbors of best_cand to pool
                a, b = best_cand
                for da, db in _NEIGH:
                    pool.add((a+da, b+db))
                improved = True
                # Break and restart loop to use updated set?
                # Or continue? Continuing might use stale degrees.
                # Let's break and restart for correctness.
                break
        
    return best, best_e

def construct():
    deadline = time.time() + 140
    n = 32
    
    # Generate a large enough candidate pool
    order = _spiral_order(10)
    
    best_pts = None
    best_e = -1
    
    # Strategy 1: Greedy from (0,0)
    pts1 = _greedy_construct(n, order)
    e1 = _count_edges(pts1)
    if e1 > best_e:
        best_e = e1
        best_pts = pts1
    
    # Polish with deterministic local search
    if time.time() < deadline - 5:
        polished1, e_polished1 = _deterministic_local_search(pts1)
        if e_polished1 > best_e:
            best_e = e_polished1
            best_pts = polished1
            
    # Strategy 2: Greedy from different starts?
    # Try a few random starts for greedy
    rng = random.Random(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        start = order[rng.randrange(len(order))]
        # Ensure start is distinct
        pts2 = _greedy_construct(n, order, start_pt=start)
        e2 = _count_edges(pts2)
        if e2 > best_e:
            best_e = e2
            best_pts = pts2
        # Polish
        if time.time() < deadline - 5:
            polished2, e_polished2 = _deterministic_local_search(pts2)
            if e_polished2 > best_e:
                best_e = e_polished2
                best_pts = polished2

    # Strategy 3: Hexagonal patch + Local Search (Fallback/Backup)
    if time.time() < deadline - 10:
        # Re-use logic from previous program if needed, but we have a new structural approach.
        # Let's just stick to Greedy + Polish.
        pass

    # Final verification
    if best_pts is None:
        # Fallback to simple spiral
        best_pts = order[:n]
        
    # Ensure distinct and within bounds
    seen = set()
    final_pts = []
    for p in best_pts:
        if p not in seen and abs(p[0]) <= 64 and abs(p[1]) <= 64:
            seen.add(p)
            final_pts.append(p)
            
    if len(final_pts) < n:
        for p in order:
            if len(final_pts) >= n:
                break
            if p not in seen and abs(p[0]) <= 64 and abs(p[1]) <= 64:
                seen.add(p)
                final_pts.append(p)
                
    return [[a,b] for (a,b) in final_pts[:n]]
