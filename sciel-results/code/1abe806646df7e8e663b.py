"""Cage solver for small k-regular graphs of girth g.
Uses LUW construction as a reliable baseline, plus random configuration model with local search."""
import time
import random
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def _elapsed(t0):
    return time.time() - t0

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    if n <= 3:
        return 3
    cand = n + 1
    if cand % 2 == 0:
        cand += 1
    while not _is_prime(cand):
        cand += 2
    return cand

def _moore_bound(k, g):
    """Lower bound on number of vertices for k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: 1 + k * sum_{i=0}^{g/2 - 2} (k-1)^i
        half = g // 2 - 1
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return 1 + k * s
    else:
        # Odd girth: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i + k*(k-1) * sum_{i=0}^{(g-5)/2} (k-1)^i
        # Simpler: 2 * sum_{i=0}^{(g-1)/2} (k-1)^i for odd? Let me use standard formula.
        # For odd g: n >= 2 * sum_{i=0}^{(g-1)/2} (k-1)^i
        half = (g - 1) // 2
        s = 0
        term = 1
        for i in range(half + 1):
            s += term
            term *= (k - 1)
        return 2 * s

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _nonback_girth(n, adj, g_limit):
    """Check if graph has no cycle shorter than g_limit using non-backtracking walks.
    Returns True if girth >= g_limit, False otherwise."""
    if n < g_limit:
        return False
    # BFS in non-backtracking state space (vertex, prev_vertex)
    # State: (current, previous). Distance = length of non-backtracking walk.
    # If we reach a state where current's neighbors include previous's previous... 
    # Actually, standard method: for each directed edge (u,v), do BFS of non-backtracking walks.
    # A cycle of length < g_limit exists iff there's a non-backtracking walk of length < g_limit 
    # that returns to the starting vertex with the correct "non-backtracking" condition.
    
    # More efficient: check all pairs at distance < g_limit in the universal cover sense.
    # Simpler approach: for each vertex, BFS in the non-backtracking state graph.
    # State: (vertex, prev_vertex) where prev_vertex is the vertex we came from (or None for start).
    # We track the shortest non-backtracking walk to each state.
    # If from state (u, v) we can go to (u, w) where w != v, and if w is the start vertex 
    # of a walk of length L-1, and L < g_limit, we have a cycle.
    
    # Actually, the cleanest check: a graph has girth >= g iff for every non-backtracking walk 
    # of length < g, it does not form a cycle. Equivalently, in the non-backtracking state space,
    # no state (u, v) is reachable from (u, v) by a non-backtracking walk of length < g 
    # that returns to the same state... this is getting complex.
    
    # Let's use the standard algorithm: for each directed edge (u,v), perform BFS in the 
    # non-backtracking state graph starting from (u,v). If we reach (v,u) with a walk of 
    # length < g-1, we have a cycle of length < g. Also check if we reach any state where 
    # the last vertex connects back to the first vertex of the walk with appropriate length.
    
    # Simpler: use the fact that girth >= g iff the graph has no cycles of length < g.
    # We can check by doing BFS from each vertex up to depth (g-1)//2 and checking for 
    # collisions in the tree. But this is O(n * (k-1)^{(g-1)/2}) which can be large.
    
    # For our purposes, let's do: for each vertex s, build the non-backtracking tree up to 
    # depth floor((g-1)/2). If any two nodes at the same depth have the same parent path... 
    # Actually, a cycle exists iff two different paths of length < g/2 lead to the same vertex.
    
    # Let's do: for each vertex s, BFS up to depth d = (g-1)//2. If we visit the same vertex 
    # twice with different parents at the same or different depths, and the sum of distances 
    # plus 2 < g, we have a cycle. This is tricky.
    
    # Alternative: use the non-backtracking matrix power approach or just check small cycles directly.
    # For g=9, we need to check cycles of length 3,4,5,6,7,8.
    
    # Let's just do a simple check: for each vertex, BFS up to depth 4 (for g=9, (g-1)/2 = 4).
    # If two BFS trees from different starting points overlap in a way that creates a cycle < g, detect it.
    
    # Actually, the most reliable simple method for small g: check all cycles of length 3 to g-1.
    # For g=9, that's cycles of length 3,4,5,6,7,8.
    # Use the standard non-backtracking walk check:
    
    # For each directed edge (s, t), do BFS in non-backtracking states up to depth g-2.
    # State: (current, prev). Start: (t, s) with distance 1.
    # If we reach state (s, x) where x != t... wait, we need to return to s.
    # If from (current, prev) we go to (next, current) where next == s and distance + 1 < g, 
    # and the path is non-backtracking, we have a cycle of length distance+1.
    
    # Let me implement this properly.
    
    visited = {}  # (vertex, prev) -> (distance, start_vertex)
    queue = deque()
    
    # For each starting directed edge
    for s in range(n):
        for t in adj[s]:
            # Start non-backtracking walk: s -> t
            state = (t, s)
            if state not in visited:
                visited[state] = 1  # distance
                queue.append((t, s, 1, s))
            else:
                # Already visited, skip for now (we'll process all)
                pass
    
    # This approach is complex. Let's use a simpler but correct method:
    # Check if there's a cycle of length < g by checking if the graph contains any 
    # C_3, C_4, ..., C_{g-1}.
    
    # For small g (like 9), we can do:
    # - C3: for each triangle, check.
    # - C4: for each pair of vertices at distance 2, check if they have a common neighbor.
    # - etc.
    
    # Let's use the standard algorithm from the literature:
    # The girth of a graph is at least g iff for every non-backtracking walk w of length 
    # l < g, the walk does not close a cycle.
    
    # Implementation:
    # For each starting vertex s, do a BFS in the non-backtracking state space.
    # State: (vertex, prev_vertex). prev_vertex is the vertex we came from.
    # Distance: length of the non-backtracking walk.
    # We track for each state the starting vertex and the distance.
    # If we reach state (s, v) with distance d < g, where s is the starting vertex, 
    # we have a cycle of length d. But we need d >= 3 and the walk to be non-backtracking.
    
    # Actually, a non-backtracking walk from s that returns to s with length d < g 
    # gives a cycle of length d (since it's non-backtracking, it's a simple cycle).
    
    # Let's implement:
    for s in range(n):
        # BFS from s in non-backtracking states
        # State: (current, prev), distance
        # Start: for each neighbor t of s, state (t, s) at distance 1
        dist = {}  # (current, prev) -> distance
        q = deque()
        for t in adj[s]:
            dist[(t, s)] = 1
            q.append((t, s, 1))
        
        while q:
            curr, prev, d = q.popleft()
            if d >= g - 1:  # No need to go further
                continue
            for nxt in adj[curr]:
                if nxt == prev:
                    continue  # Non-backtracking
                if nxt == s and d + 1 < g:
                    return False  # Found a cycle of length d+1 < g
                new_state = (nxt, curr)
                if new_state not in dist:
                    dist[new_state] = d + 1
                    q.append((nxt, curr, d + 1))
    
    return True

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    t0 = time.time()
    deadline = t0 + 100.0

    if k == 4:
        q = 4
    else:
        q = _next_prime(k)

    best_d = None
    for d in range(3, 30):
        girth = d + 5 if d % 2 == 1 else d + 4
        if girth >= g:
            best_d = d
            break
    if best_d is None:
        best_d = g - 4 if (g - 4) % 2 == 1 else g - 3
    d = best_d

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    vi = 0
    while vi < len(order):
        vertex = order[vi]
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
        vi += 1

        if time.time() > deadline:
            break

    n = len(order)
    return n, edges

def _random_config(k, g, n_target, deadline):
    """Try to construct a k-regular graph of girth >= g with approximately n_target vertices
    using random configuration model with local search."""
    if n_target % 2 != 0:
        n_target += 1
    n = n_target
    
    t0 = time.time()
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random configuration model
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        random.shuffle(stubs)
        
        # Pair up stubs
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if (min(u, v), max(u, v)) in edges:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
        
        if not valid:
            continue
        
        adj = _build_adj(n, list(edges))
        
        # Check degrees
        deg_ok = True
        for v in range(n):
            if len(adj[v]) != k:
                deg_ok = False
                break
        if not deg_ok:
            continue
        
        # Check girth
        if _nonback_girth(n, adj, g):
            return n, list(edges)
    
    return None

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 150s kill
    
    mb = _moore_bound(k, g)
    
    # Try LUW construction
    luw_result = _luw(k, g)
    n_luw, edges_luw = luw_result
    
    # Verify LUW result
    if n_luw >= mb and n_luw < 100000:
        adj = _build_adj(n_luw, edges_luw)
        deg_ok = all(len(adj[v]) == k for v in range(n_luw))
        if deg_ok and _nonback_girth(n_luw, adj, g):
            return n_luw, edges_luw
    
    # Try random configuration model with various sizes
    for multiplier in [1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
        n_target = int(mb * multiplier)
        if n_target < mb:
            n_target = mb
        result = _random_config(k, g, n_target, deadline)
        if result is not None:
            n, edges = result
            if n < n_luw:
                return n, edges
        if time.time() > deadline - 10:
            break
    
    return n_luw, edges_luw

# EVOLVE-BLOCK-END
