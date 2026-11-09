import time
import math
import random
import heapq

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

def _build_greedy(n, max_hex_dist=10):
    """
    Build shape by greedily adding the point with the most neighbors in the current set.
    Tie-breakers: smallest hex distance, then angle.
    """
    if n <= 0: return []
    current_set = set()
    current_list = []
    # Start with origin
    current_set.add((0,0))
    current_list.append((0,0))
    
    # Priority queue items: (-degree, hex_dist, angle, a, b)
    # We want max degree, so we negate it for min-heap.
    # We want min hex_dist.
    # We want consistent angle ordering.
    
    pq = []
    seen_candidates = set()
    
    def _push_neighbor(a, b):
        for da, db in _NEIGH:
            na, nb = a+da, b+db
            if (na, nb) not in current_set and (na, nb) not in seen_candidates:
                if _hexdist(na, nb) <= max_hex_dist:
                    deg = _deg((na, nb), current_set)
                    x = na + 0.5*nb
                    y = 0.8660254037844386*nb
                    ang = math.atan2(y, x)
                    heapq.heappush(pq, (-deg, _hexdist(na, nb), ang, na, nb))
                    seen_candidates.add((na, nb))

    _push_neighbor(0, 0)
    
    while len(current_list) < n:
        if not pq:
            break
        # Pop best candidate
        neg_deg, hd, ang, a, b = heapq.heappop(pq)
        # It might have been added to set already? No, we check seen_candidates.
        # But its degree might have changed? 
        # Actually, the degree of a candidate only increases as we add more points.
        # So the priority might be stale. 
        # However, for this small problem, re-evaluating is safer or we just trust the heap.
        # If we trust the heap, we might pick a point that had degree 2 when pushed, but now has degree 3.
        # Another point with degree 3 might be lower in the heap if it was pushed earlier with degree 2?
        # To be precise, we should re-calculate degree or use a dynamic update.
        # Given n=32, we can afford to be slightly less efficient or just re-calc.
        
        # Let's just re-calculate the degree of the popped point to be sure it's still the best?
        # No, that defeats the purpose. 
        # Standard approach: Just pop. If the degree is stale, it's still a valid move.
        # For "most neighbors", stale info means we might pick a point that *used to be* best.
        # This is usually fine for growth.
        
        # Add to set
        current_set.add((a,b))
        current_list.append((a,b))
        
        # Update neighbors
        _push_neighbor(a, b)
        
        # Also, the degree of other candidates in the heap might have increased.
        # We don't update them in the heap. This is a known limitation of simple greedy heaps.
        # But for compact shapes, the "front" moves outward, and degrees of boundary points increase.
        # This should still work well.
        
    return current_list

def _try_improve(pts, max_time=8.0, seed=42):
    """Local search to improve edge count via single-point swaps with incremental delta."""
    rng = random.Random(seed)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build candidate pool - slightly larger for better escape
    pool = set()
    for (a,b) in best:
        for r in range(0, 7):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 14:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 10000
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

def _try_improve_2swap(pts, max_time=5.0, seed=123):
    """Local search with 2-point swaps to escape local optima."""
    rng = random.Random(seed)
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build candidate pool
    pool = set()
    for (a,b) in best:
        for r in range(0, 7):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 14:
                        pool.add((a2,b2))
    pool = list(pool)
    
    start = time.time()
    it = 0
    stall = 0
    max_stall = 20000
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
        lost = _deg(cur1, s) + _deg(cur2, s)
        # Check if cur1 and cur2 are adjacent
        a1, b1 = cur1
        a2, b2 = cur2
        for da, db in _NEIGH:
            if (a1+da, b1+db) == cur2:
                lost -= 1
                break
        
        s2 = s - {cur1, cur2}
        gained = _deg(cand1, s2) + _deg(cand2, s2)
        a1, b1 = cand1
        a2, b2 = cand2
        for da, db in _NEIGH:
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
    order = _spiral_order(14)
    spiral_pts = order[:n]
    e_spiral = _count_edges(spiral_pts)
    
    # Strategy 3: Greedy Growth
    greedy_pts = _build_greedy(n)
    e_greedy = _count_edges(greedy_pts)
    
    # Pick the best starting point
    best = hex_patch
    best_e = e_hex
    if e_spiral > best_e:
        best, best_e = spiral_pts, e_spiral
    if e_greedy > best_e:
        best, best_e = greedy_pts, e_greedy
    
    # Local search from multiple starting points with different seeds
    candidates = [hex_patch, spiral_pts, greedy_pts]
    
    results = []
    for idx, start_pts in enumerate(candidates):
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        # Single swap search
        t1 = min(remaining * 0.4, 10.0)
        best1, e1 = _try_improve(start_pts, max_time=t1, seed=42 + idx)
        results.append((e1, best1))
        
        # 2-swap search
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        t2 = min(remaining * 0.3, 8.0)
        best2, e2 = _try_improve_2swap(start_pts, max_time=t2, seed=123 + idx)
        results.append((e2, best2))
    
    # Pick the best result
    for e, pts in results:
        if e > best_e:
            best_e = e
            best = pts
    
    # Final polish: more iterations on the best solution
    remaining = deadline - time.time()
    if remaining > 3.0:
        # Single swap polish
        t3 = min(remaining * 0.5, 20.0)
        best3, e3 = _try_improve(best, max_time=t3, seed=999)
        if e3 > best_e:
            best_e = e3
            best = best3
        
        # 2-swap polish
        remaining = deadline - time.time()
        if remaining > 2.0:
            t4 = min(remaining * 0.4, 15.0)
            best4, e4 = _try_improve_2swap(best, max_time=t4, seed=2024)
            if e4 > best_e:
                best_e = e4
                best = best4
    
    # Verify and return
    # Ensure all points are distinct and within bounds
    seen = set()
    final_pts = []
    for (a,b) in best:
        if (a,b) not in seen and abs(a) <= 64 and abs(b) <= 64:
            seen.add((a,b))
            final_pts.append((a,b))
    
    # If we somehow have fewer than 32 points, fill with spiral points
    if len(final_pts) < n:
        for (a,b) in order:
            if len(final_pts) >= n:
                break
            if (a,b) not in seen and abs(a) <= 64 and abs(b) <= 64:
                seen.add((a,b))
                final_pts.append((a,b))
    
    return [[a,b] for (a,b) in final_pts[:n]]
