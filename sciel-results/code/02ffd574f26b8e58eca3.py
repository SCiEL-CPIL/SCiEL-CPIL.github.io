import math
import time

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _build_conflict_adj(n):
    """Build adjacency as list of frozensets of conflicting indices (0-based for values 1..n)."""
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i, n):
            b = j + 1
            if a + b >= 4 and _is_square(a + b):
                adj[i].add(j)
                adj[j].add(i)
    return [frozenset(s) for s in adj]

def _greedy_color_bound(adj, candidates):
    """
    Greedy coloring upper bound for maximum independent set.
    candidates: set of vertex indices still available.
    Returns upper bound on MIS size in subgraph induced by candidates.
    """
    if not candidates:
        return 0
    # Order by degree in subgraph (descending) for tighter bound
    sub = list(candidates)
    sub.sort(key=lambda v: len(adj[v] & candidates), reverse=True)
    colors = {}
    color_count = 0
    for v in sub:
        # Find smallest available color
        used = set()
        for u in adj[v]:
            if u in candidates and u in colors:
                used.add(colors[u])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        color_count = max(color_count, c + 1)
    return color_count

def _local_search(adj, initial_set):
    """Local search: try to add elements, then try swaps."""
    S = set(initial_set)
    improved = True
    while improved:
        improved = False
        # Try adding
        for v in range(len(adj)):
            if v not in S:
                if not adj[v] & S:
                    S.add(v)
                    improved = True
        # Try swaps: remove one, add one or more
        for v in list(S):
            S_remove = S - {v}
            added = 0
            for u in range(len(adj)):
                if u not in S_remove and not adj[u] & S_remove:
                    S_remove.add(u)
                    added += 1
            if added >= 1:
                S = S_remove
                improved = True
                break
    return S

def _branch_and_bound(adj, deadline):
    """Branch and bound for maximum independent set."""
    n = len(adj)
    best_size = 0
    best_set = set()
    
    # Precompute adjacency as bitmasks for speed
    adj_mask = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        adj_mask[i] = m
    
    # Get initial solution via greedy
    candidates = set(range(n))
    greedy = set()
    while candidates:
        # Pick vertex with min degree in remaining
        v = min(candidates, key=lambda x: len(adj[x] & candidates))
        greedy.add(v)
        candidates -= adj[v]
        candidates.discard(v)
    
    best_size = len(greedy)
    best_set = set(greedy)
    
    # Try to improve with local search
    ls_set = _local_search(adj, best_set)
    if len(ls_set) > best_size:
        best_size = len(ls_set)
        best_set = ls_set
    
    # B&B search
    def popcount(x):
        return bin(x).count('1')
    
    def color_bound_from_mask(mask):
        """Greedy coloring bound on vertices in mask."""
        if mask == 0:
            return 0
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        # Sort by degree in subgraph
        verts.sort(key=lambda v: popcount(adj_mask[v] & mask), reverse=True)
        colors = {}
        max_c = 0
        for v in verts:
            used = set()
            neighbors_in_mask = adj_mask[v] & mask
            nm = neighbors_in_mask
            while nm:
                lsb = nm & -nm
                idx = lsb.bit_length() - 1
                if idx in colors:
                    used.add(colors[idx])
                nm ^= lsb
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            max_c = max(max_c, c + 1)
        return max_c
    
    def branch(P, X):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        if P == 0:
            if len(X) > best_size:
                best_size = len(X)
                best_set = set(X)
            return
        bound = color_bound_from_mask(P)
        if len(X) + bound <= best_size:
            return
        # Select vertex with max degree in P
        verts = []
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        if len(verts) > 1:
            verts.sort(key=lambda v: popcount(adj_mask[v] & P), reverse=True)
        v = verts[0]
        
        # Include v
        P_inc = P & ~(1 << v) & ~adj_mask[v]
        X.append(v)
        branch(P_inc, X)
        X.pop()
        
        # Exclude v
        P_exc = P & ~(1 << v)
        branch(P_exc, X)
    
    branch((1 << n) - 1, [])
    
    return best_set

def construct():
    start = time.time()
    deadline = start + 75  # 15s safety margin
    
    n = 256
    adj = _build_conflict_adj(n)
    
    # Check deadline
    if time.time() > deadline:
        return list(range(1, 257, 3))
    
    best_set = _branch_and_bound(adj, deadline)
    
    if not best_set:
        return list(range(1, 257, 3))
    
    # Convert 0-based indices to 1-based values
    result = sorted(v + 1 for v in best_set)
    
    # Verify
    valid = True
    s = set(result)
    for a in s:
        for b in s:
            if a <= b and _is_square(a + b):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return list(range(1, 257, 3))
    
    return result
