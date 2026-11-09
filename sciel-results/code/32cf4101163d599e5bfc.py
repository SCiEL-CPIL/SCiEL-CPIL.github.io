import math
import time

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

def _perimeter(pts):
    """Count boundary edges (edges from a point in pts to a neighbor NOT in pts)."""
    s = set(pts)
    per = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) not in s:
                per += 1
    return per

def _local_search(pts, max_iter=2000, deadline=None):
    """Swap-based local search: replace one point with another to increase edge count."""
    pts_set = set(pts)
    best_edges = _count_edges(pts_set)
    # Candidate pool: hexagonal region
    R = 6
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if _hexdist(a,b) <= R:
                cands.append((a,b))
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        for i, (pa, pb) in enumerate(pts_set):
            for ca, cb in cands:
                if (ca, cb) in pts_set:
                    continue
                # Try swap
                new_set = pts_set - {(pa,pb)} | {(ca,cb)}
                e = _count_edges(new_set)
                if e > best_edges:
                    pts_set = new_set
                    best_edges = e
                    improved = True
                    break
            if improved:
                break
    return list(pts_set), best_edges

def construct():
    n = 32
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Phase 1: Submodular greedy with perimeter-aware scoring
    # Start with center
    R_search = 5
    all_cands = []
    for a in range(-R_search, R_search+1):
        for b in range(-R_search, R_search+1):
            if _hexdist(a,b) <= R_search:
                all_cands.append((a,b))
    
    # Greedy: pick point that maximizes (new_edges - alpha * perimeter_growth)
    # Equivalent to minimizing perimeter growth for a given edge gain
    alpha = 0.1
    selected = [(0,0)]
    sel_set = {(0,0)}
    
    for _ in range(n - 1):
        if time.time() > deadline:
            break
        best_score = -1
        best_pt = None
        for (a,b) in all_cands:
            if (a,b) in sel_set:
                continue
            # Count new edges this point would add
            new_edges = 0
            for da, db in _NEIGH:
                if (a+da, b+db) in sel_set:
                    new_edges += 1
            # Perimeter change: new boundary edges minus edges that become internal
            # A new point has 6 boundary edges minus new_edges (which connect to interior)
            # But it also removes boundary edges from existing points that connect to it
            # Net perimeter change = (6 - new_edges) - new_edges = 6 - 2*new_edges
            per_change = 6 - 2 * new_edges
            score = new_edges - alpha * per_change
            if score > best_score:
                best_score = score
                best_pt = (a,b)
        if best_pt is None:
            break
        selected.append(best_pt)
        sel_set.add(best_pt)
    
    best_pts = list(selected)
    best_edges = _count_edges(set(best_pts))
    
    # Phase 2: Local search to polish
    if time.time() < deadline - 5:
        polished, e2 = _local_search(best_pts, max_iter=5000, deadline=deadline)
        if e2 > best_edges:
            best_pts = polished
            best_edges = e2
    
    # Ensure exactly 32 points (should be, but safety check)
    if len(best_pts) < n:
        # Fill with spiral points
        R = 3
        cands2 = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if _hexdist(a,b) <= R:
                    hd = _hexdist(a,b)
                    x = a + 0.5*b
                    y = 0.8660254037844386*b
                    ang = math.atan2(y, x)
                    cands2.append((hd, ang, a, b))
        cands2.sort()
        existing = set(best_pts)
        for hd, ang, a, b in cands2:
            if len(best_pts) >= n:
                break
            if (a,b) not in existing:
                best_pts.append((a,b))
                existing.add((a,b))
    
    return [[a,b] for (a,b) in best_pts[:n]]
