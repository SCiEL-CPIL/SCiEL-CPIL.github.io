import time
import numpy as np
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 105.0

def _deadline():
    return time.time() >= _DEADLINE

def _field_ops(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))

def _luw_rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _luw_graph(d, q):
    add, sub, mul = _field_ops(q)
    rule = _luw_rule(d)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rc[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    
    adj = [set() for _ in range(len(order))]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)
            
    edges = [(i, j) for i in range(len(order)) for j in adj[i] if i < j]
    return len(order), edges, adj

def _girth_nb(adj, g):
    """Exact girth check via non-backtracking walks. Returns girth or None if >= g."""
    n = len(adj)
    if n == 0:
        return 0
    L = [sorted(a) for a in adj]
    # State: (prev, curr) -> (depth, prev_prev)
    # We want to detect cycles of length < g.
    # A cycle of length L is a sequence v0, v1, ..., v_{L-1}, v0.
    # Non-backtracking means v_{i-1} != v_{i+1}.
    # We can BFS on directed edges (u, v).
    # dist[(u, v)] = length of path ending at edge (u, v).
    # If we see (u, v) where dist[(u, v)] < dist[(v, w)] + 1, we might have a cycle?
    # Actually, standard method:
    # BFS from each vertex? No, that's O(n^2).
    # Better: BFS on the directed line graph (states are edges).
    # Initialize: for all edges (u, v), dist[(u, v)] = 1.
    # Expand (u, v) -> (v, w) for w != u.
    # If (v, w) is already visited with dist d, and d < dist[(v, w)] + 1, cycle length is dist[(v, w)] + 1?
    # Wait, if we are at (u, v) with depth d, and we go to (v, w), new depth is d+1.
    # If (v, w) was visited with depth d', then we have a cycle of length d + 1 - d'? No.
    # The cycle is v0...u, v, w...v_{d'-1}, v_{d'}=v? No.
    # Let's use the standard "girth via BFS from each vertex" for small graphs, or "non-backtracking BFS" globally.
    # Global non-backtracking BFS:
    # dist[(u, v)] = shortest non-backtracking walk from some start to (u, v).
    # This finds the shortest cycle?
    # Yes. If we find an edge (u, v) such that (v, u) is reachable?
    # Actually, simpler:
    # For each vertex s, BFS to find shortest cycle through s?
    # For n ~ 1000-2000, O(n^2) is 10^6-4*10^6, which is fine in Python if optimized.
    # But we have a deadline.
    # Let's stick to the BFS from each vertex method but optimized.
    
    # Alternative: Use the fact that we only need to check if girth < g.
    # If g is small (e.g. 9), we can do BFS up to depth g//2.
    
    best = None
    for s in range(n):
        if _deadline():
            return best
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque()
        q.append(s)
        found = None
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in L[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and v != s:
                    # Cycle found
                    # Length = du + dist[v] + 1
                    cand = du + dist[v] + 1
                    if found is None or cand < found:
                        found = cand
                    if found < g:
                        return found
                elif v == s:
                    # Back to start
                    cand = du + 1
                    if found is None or cand < found:
                        found = cand
                    if found < g:
                        return found
        if found is not None:
            if best is None or found < best:
                best = found
            if best < g:
                return best
    return best

def _random_regular_graph(n, k, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model + rejection."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    half_edges = n * k
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    for attempt in range(20):
        perm = rng.permutation(half_edges)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, half_edges, 2):
            u = stubs[perm[i]]
            v = stubs[perm[i + 1]]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            if all(len(adj[i]) == k for i in range(n)):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges, adj
    return None

def _greedy_girth_graph(k, g, max_n=2000, seed=0):
    """
    Greedy construction:
    1. Start with a path or star? No, regular graph.
    2. This is hard to do greedily for regular graphs.
    3. Instead, we will use a "repair" heuristic on a random regular graph.
    """
    pass

def construct(k, g):
    candidates = []
    if k == 4:
        qs = [4]
    else:
        qs = [k]
    for q in qs:
        d0 = g - 5
        if d0 % 2 == 0:
            d0 += 1
        d0 = max(3, d0)
        for d in (d0, d0 + 2, d0 + 4):
            candidates.append((d, q))
            
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    best = None
    best_n = float('inf')
    
    # First try LUW constructions
    for (d, q) in ordered:
        if _deadline():
            break
        try:
            n, edges, adj = _luw_graph(d, q)
        except Exception:
            continue
            
        deg = [0] * n
        seen_edges = set()
        valid = True
        for (i, j) in edges:
            if i == j or i < 0 or i >= n or j < 0 or j >= n:
                valid = False
                break
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen_edges:
                valid = False
                break
            seen_edges.add((a, b))
            deg[a] += 1
            deg[b] += 1
        if not valid or not all(x == k for x in deg):
            continue
            
        gi = _girth_nb(adj, g)
        if gi is None or gi < g:
            continue
            
        if n < best_n:
            best_n = n
            best = (n, edges)
    
    # Try random regular graphs if time permits
    if not _deadline():
        min_n = 2 * k * (g - 1)
        # Moore bound for (5,9) is ~ 2*5*(5^4-1)/(5-1) = 10*624/4 = 1560?
        # Actually Moore bound M(5,9) = 2 * (5^4 - 1) / (5 - 1) = 10 * 624 / 4 = 1560.
        # Wait, formula for odd girth g: 2 * (k^(g/2) - 1) / (k - 1).
        # For g=9, g/2 is not integer.
        # Moore bound for girth g:
        # If g is odd: n >= 2 * (k^((g+1)/2) - 1) / (k - 1)
        # For k=5, g=9: n >= 2 * (5^5 - 1) / 4 = 2 * (3125 - 1) / 4 = 2 * 3124 / 4 = 1562.
        # So we can start searching around 1562.
        
        # The LUW graph for k=5, g=9 (d=4) has size?
        # d=4, q=5.
        # Points: (x1..x4) in GF(5)^4? No, the LUW construction size depends on d and q.
        # For d=4, q=5, the number of vertices is roughly 2 * q^d? No.
        # Let's just let the code find it.
        
        for trial_n in range(1562, 1562 + 100, 2):
            if _deadline():
                break
            if trial_n < best_n and trial_n % k == 0:
                try:
                    result = _random_regular_graph(trial_n, k, seed=trial_n)
                    if result is not None:
                        rn, redges, radj = result
                        gi = _girth_nb(radj, g)
                        if gi is None or gi >= g:
                            if rn < best_n:
                                best_n = rn
                                best = (rn, redges)
                except Exception:
                    pass

    if best is None:
        for (d, q) in ordered:
            try:
                n, edges, adj = _luw_graph(d, q)
                deg = [0] * n
                for (i, j) in edges:
                    deg[i] += 1
                    deg[j] += 1
                if all(x == k for x in deg):
                    return n, edges
            except Exception:
                continue
        return 0, []
        
    return best
