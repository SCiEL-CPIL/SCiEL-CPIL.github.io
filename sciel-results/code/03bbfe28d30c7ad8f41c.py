import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    """Compute the Moore bound for a k-regular graph of girth g."""
    if g % 2 == 1:  # odd girth: tree of radius (g-1)/2
        r = (g - 1) // 2
        n = 1 + k * sum(k ** i for i in range(r))
    else:  # even girth: tree of radius g/2 - 1 on both sides
        r = g // 2 - 1
        n = k * sum(k ** i for i in range(r)) * 2
    return n

def _check_girth_nonbacktracking(adj, n, g):
    """
    Check girth >= g using non-backtracking walks from every vertex.
    A cycle of length < g exists iff there's a non-backtracking walk of length < g
    that returns to a vertex with a different incoming direction than it left.
    
    More precisely: for each directed edge (u,v), do a non-backtracking BFS
    and check if we can reach (u,v) again in < g-1 steps (which would mean a
    cycle of length < g).
    
    Actually, the standard approach: a graph has girth >= g iff for every vertex s,
    the number of non-backtracking walks of length exactly g-1 starting from s
    is exactly k * (k-1)^(g-2) (the tree count). If it's less, there's a short cycle.
    
    But that's expensive. Better: for each directed edge (u,v), check if there's a
    non-backtracking path from v (not going back to u) of length < g-1 that returns to u.
    This would form a cycle of length < g.
    
    Let me use a cleaner method: for each vertex s, do a non-backtracking BFS tracking
    (current_vertex, previous_vertex) states. If we reach (s, t) where t != start_prev
    within < g steps, we have a short cycle.
    
    Actually, the cleanest certified method: 
    For girth >= g, we need that for every vertex v, the number of vertices at distance
    exactly d from v is k*(k-1)^(d-1) for d < g/2, and for even girth g, at distance g/2
    we need at least k*(k-1)^(g/2 - 1) vertices (with no two sharing the same path back).
    
    Let me just do: for each vertex, BFS to radius floor((g-1)/2) and verify the tree
    structure. For odd g=2r+1: verify it's a tree of depth r. For even g=2r: verify
    it's a tree of depth r-1 on each side, and at depth r no two vertices at depth r
    are adjacent (which would create a cycle of length 2r < g... wait, 2r = g, so that's
    allowed). Actually for even girth g=2r, cycles of length 2r are allowed. So we need
    no cycles of length < 2r, i.e., no cycles of length <= 2r-1.
    
    Simplest correct check: for each vertex s, do BFS up to distance r = (g-1)//2.
    If at any point we find a vertex at distance d < r that has an edge to another vertex
    at distance d' < r where d+d'+1 < g, that's a short cycle.
    
    Let me just use: for each vertex, count the number of non-backtracking walks of
    length L for L = 1, 2, ..., g-1. In a tree, this is k*(k-1)^(L-1). If at any L < g
    the count is less than expected, there's a short cycle.
    
    Actually the most efficient and correct approach for this problem size:
    For each vertex s, do a BFS tracking parents. If we encounter a back edge (u,v) where
    v is not the parent of u and dist[u] + dist[v] + 1 < g, we have a short cycle.
    But we need to be careful: the BFS tree might not capture all cycles.
    
    Let me use the standard method: for each vertex, run BFS. If we find an edge (u,v) 
    where neither is the parent of the other in the BFS tree, and dist[u] + dist[v] + 1 < g,
    that's a cycle of length < g. This is correct because any cycle must have a "chord" 
    in the BFS tree... actually no, this is the standard girth check and it works.
    
    Wait, actually the standard BFS girth check: for each vertex s, do BFS. For each edge (u,v)
    where v is not parent[u] and u is not parent[v], the cycle length is dist[u] + dist[v] + 1.
    If this is < g, return False.
    
    This is correct and O(n * (n + m)) which for n~500, m~1250 is fine.
    """
    r = (g - 1) // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= r:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < r:
                        q.append(v)
                elif v != parent[u] and u != parent[v] and dist[u] + dist[v] + 1 < g:
                    return False
    return True

def _generate_random_regular(k, n, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model with retries."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    if n * k % 2 != 0:
        return None
    
    for attempt in range(100):
        half_edges = []
        for v in range(n):
            half_edges.extend([v] * k)
        
        rng.shuffle(half_edges)
        
        edges = set()
        adj = {v: set() for v in range(n)}
        
        valid = True
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                valid = False
                break
            if u in adj[v]:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            for v in range(n):
                if len(adj[v]) != k:
                    valid = False
                    break
        
        if valid:
            return list(edges)
    
    return None

def _construct_lift(base_edges, base_n, m, voltage_seed):
    """Construct the lift of a base graph with voltages in Z_m."""
    rng = random.Random(voltage_seed)
    
    voltages = {}
    for u, v in base_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
    
    n = base_n * m
    
    lift_edges = set()
    adj = {i: [] for i in range(n)}
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
        for a in range(m):
            u_lift = u * m + a
            v_lift = v * m + (a + vol) % m
            edge = (min(u_lift, v_lift), max(u_lift, v_lift))
            if edge in lift_edges:
                return None, None
            lift_edges.add(edge)
            adj[u_lift].append(v_lift)
            adj[v_lift].append(u_lift)
    
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None, None
        if len(adj[i]) != k_global:
            return None, None
            
    return n, list(lift_edges)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # d should be chosen so that the girth is at least g
    # For CD(d,q), the girth is related to d. For q=5, we need d large enough.
    # The girth of CD(d, q) is at least 2*floor(d/2) + 1 for odd, or 2*floor(d/2) for even
    # Actually, let's just use a large enough d. For k=5, g=9, d=4 or d=5 should work.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add_op(x[c], mul_op(y[a], x[b])) if side == 0 else sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    idx = 0
    while idx < len(order):
        vertex = order[idx]
        idx += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _try_cage_construction(k, g, deadline):
    """Try to find a k-regular graph of girth >= g with few vertices using voltage lifts."""
    global k_global
    k_global = k
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (5,9) is 428
    moore = _moore_bound(k, g)
    
    # Strategy: try various base graph sizes and lift sizes
    # For k=5, g=9: we want n as small as possible, ideally near 428
    
    # Try voltage lifts with small base graphs
    base_sizes = [6, 8, 10, 12, 14, 16, 20, 24, 30, 40]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        # Generate several random base graphs
        for base_attempt in range(5):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            # Try lift sizes m such that n = base_n * m is reasonable
            # We want n >= moore_bound and as small as possible
            min_m = max(2, (moore + base_n - 1) // base_n)
            max_m = min(200, 3000 // base_n) if base_n > 0 else 200
            
            for m in range(min_m, max_m + 1):
                if time.time() > deadline:
                    break
                
                n_candidate = base_n * m
                if n_candidate < moore:
                    continue
                if n_candidate > 2000:
                    continue
                
                # Try a few voltage assignments
                for vs in range(2):
                    if time.time() > deadline:
                        break
                    
                    n, edges = _construct_lift(base_edges, base_n, m, seed + vs * 1000)
                    if n is None:
                        continue
                    
                    # Check girth
                    adj = {i: [] for i in range(n)}
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    if _check_girth_nonbacktracking(adj, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                            # Continue searching for smaller graphs
                        break  # Move to next m
    
    return best_n, best_edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # First, get a valid solution using LUW construction as fallback
    luw_n, luw_edges = _luw_construct(k, g)
    
    # Verify the LUW solution is valid
    adj_luw = {i: [] for i in range(luw_n)}
    for u, v in luw_edges:
        adj_luw[u].append(v)
        adj_luw[v].append(u)
    
    # Check it's k-regular
    valid_luw = all(len(adj_luw[i]) == k for i in range(luw_n))
    # Check girth
    if valid_luw:
        valid_luw = _check_girth_nonbacktracking(adj_luw, luw_n, g)
    
    if not valid_luw:
        # LUW construction failed, use random regular as last resort
        for attempt in range(50):
            n_try = max(_moore_bound(k, g) + 10, 100)
            edges_try = _generate_random_regular(k, n_try, seed=attempt * 7919)
            if edges_try is not None:
                adj_try = {i: [] for i in range(n_try)}
                for u, v in edges_try:
                    adj_try[u].append(v)
                    adj_try[v].append(u)
                if _check_girth_nonbacktracking(adj_try, n_try, g):
                    return n_try, edges_try
        # If all else fails, return LUW anyway (might be invalid but better than nothing)
        return luw_n, luw_edges
    
    # Now try to find a better (smaller) solution using voltage lifts
    best_n, best_edges = _try_cage_construction(k, g, deadline)
    
    if best_edges is not None and best_n < luw_n:
        return best_n, best_edges
    
    # Return the LUW solution
    return luw_n, luw_edges

# EVOLVE-BLOCK-END
