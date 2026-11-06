import time

_START = None
_DEADLINE = None

def _now():
    return time.time()

def _hexdist(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return (abs(da) + abs(da + db) + abs(db)) // 2

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    if len(Sset) != 32:
        return False
    return all(abs(a) <= 64 and abs(b) <= 64 for a, b in Sset)

def construct():
    global _START, _DEADLINE
    if _START is None:
        _START = time.time()
        _DEADLINE = _START + 110.0

    # Generate candidate pool: hexagon of radius 3
    # This contains all points with hexdist <= 3
    pool = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 3:
                pool.append((a, b))
    
    poolset = set(pool)
    
    # We need 32 points.
    # Points with hexdist <= 2 are definitely in the optimal core (19 points).
    # We need 13 more from hexdist == 3.
    
    # Sort pool by hexdist
    def sort_key(p):
        return _hexdist(p, (0,0))
        
    sorted_pool = sorted(pool, key=sort_key)
    
    # Separate into inner (dist <= 2) and outer (dist == 3)
    inner = [p for p in sorted_pool if _hexdist(p, (0,0)) <= 2]
    outer = [p for p in sorted_pool if _hexdist(p, (0,0)) == 3]
    
    # We already have 19 points. Need 13 more.
    # To maximize edges, pick outer points that have the most neighbors in the current set (inner + already picked outer).
    # However, since we are picking from a static set, we can just count neighbors in `inner` first.
    # Points in `outer` that are adjacent to more points in `inner` are better.
    
    def neighbor_count_in_inner(p):
        return sum(1 for n in _neighbors(p) if n in set(inner))
        
    # Sort outer by neighbor count in inner (descending)
    # Tie-breaker: arbitrary, but consistent
    outer_sorted = sorted(outer, key=lambda p: -neighbor_count_in_inner(p))
    
    selected = inner + outer_sorted[:13]
    Sset = set(selected)
    
    # Verify validity
    if not _valid(Sset):
        # Fallback to a simple grid if something goes wrong (should not happen)
        Sset = set([(i, j) for i in range(16) for j in range(2)])
        
    # Calculate edges to ensure we hit 76
    # If not 76, we could run a quick local search, but the deterministic selection 
    # of the most connected boundary points should yield the optimal compact shape.
    
    # Just in case the deterministic heuristic misses the absolute global optimum 
    # (which is unlikely for this specific n=32 case, but possible if tie-breaking is wrong),
    # we can run a very fast local search to polish.
    
    # Define swap candidates for local search: points in pool not in Sset
    # Actually, for speed and reliability, we'll just do a few passes of local search 
    # if the edge count is not 76.
    
    edges = _edge_count(Sset)
    
    if edges < 76:
        # Simple local search improvement
        # Try swapping points in Sset with points in pool \ Sset
        swap_cand = {p: [q for q in pool if q != p and _hexdist(p, q) <= 2] for p in pool}
        
        S = list(Sset)
        Sset_local = set(S)
        cur = edges
        improved = True
        passes = 0
        while improved and passes < 10 and _now() < _DEADLINE - 1.0:
            improved = False
            passes += 1
            for i in range(len(S)):
                p = S[i]
                for q in swap_cand[p]:
                    if q in Sset_local:
                        continue
                    newS = set(Sset_local)
                    newS.discard(p)
                    newS.add(q)
                    ne = _edge_count(newS)
                    if ne > cur:
                        Sset_local = newS
                        S[i] = q
                        cur = ne
                        improved = True
                        break
                if improved:
                    break
        Sset = Sset_local
        edges = cur
        
    # Final result
    best = [list(t) for t in Sset]
    
    # Sanity checks
    assert len(best) == 32
    assert len(set(tuple(p) for p in best)) == 32
    assert all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in best)
    
    return best
