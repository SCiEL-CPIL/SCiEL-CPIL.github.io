import time
import random
import numpy as np
from collections import deque

def _build_luw(k, g):
    """Construct Lazebnik-Ustimenko-Woldar graph as fallback."""
    q = k
    if q == 4:
        d = 5 if g <= 10 else 6
        tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def _add(a,b): return a ^ b
        def _sub(a,b): return a ^ b
        def _mul(a,b): return tbl[a][b]
    else:
        d = max(3, g - 5 if (g-5) % 2 else g - 4)
        def _add(a,b): return (a+b) % q
        def _sub(a,b): return (a-b) % q
        def _mul(a,b): return (a*b) % q
    
    rules = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        rules[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def _nbrs(v):
        side, x = v
        for f in range(q):
            y = [None, f] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rules[c]
                if side == 0:
                    y[c] = _add(x[c], _mul(y[a], x[b]))
                else:
                    y[c] = _sub(x[c], _mul(x[a], y[b]))
            yield (1-side, tuple(y))
    
    start = (0, (None,) + (0,)*d)
    idx = {start: 0}
    ordered = [start]
    edge_list = []
    for v in ordered:
        for nb in _nbrs(v):
            if nb not in idx:
                idx[nb] = len(ordered)
                ordered.append(nb)
            if v[0] == 0:
                edge_list.append((idx[v], idx[nb]))
    return len(ordered), edge_list

def _gen_4reg(n):
    """Generate a random simple 4-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    return adj

def _fast_girth_check(adj, n, g):
    """Fast girth check: for each vertex, BFS to find if any cycle < g exists.
    Returns True if girth >= g, False otherwise."""
    for s in range(n):
        # BFS from s, tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    dq.append(v)
                elif v != parent[u] and dist[v] <= du:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _verify_fast(n, edges, k, g):
    """Fast verification: simple, k-regular, connected, girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False
    # girth check
    if not _fast_girth_check(adj, n, g):
        return False
    return True

def construct(k, g):
    """Construct k-regular graph with girth at least g, minimizing vertices."""
    
    # For k=4, g=10: try to find a graph with n=320 (known optimal)
    if k == 4 and g == 10:
        t0 = time.time()
        hard_limit = t0 + 115  # Leave 5s buffer for 120s limit
        
        # Strategy: try random 4-regular graphs on 320 vertices with repair
        # The known optimal is 320, so focus there first
        best_result = None
        
        # First, try to build a 320-vertex graph
        for n in [320, 322, 324, 326, 328, 330]:
            if time.time() > hard_limit - 10:
                break
            
            n_deadline = time.time() + 15  # 15 seconds per n
            attempts = 0
            while time.time() < n_deadline and attempts < 200:
                attempts += 1
                adj = _gen_4reg(n)
                if adj is None:
                    continue
                
                # Quick girth check
                if _fast_girth_check(adj, n, 10):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if _verify_fast(n, edges, 4, 10):
                        return n, edges
                
                # Repair: find short cycles and try to break them via swaps
                for step in range(50):
                    if time.time() > n_deadline:
                        break
                    
                    # Find a short cycle
                    found_cycle = False
                    for s in range(n):
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[s] = 0
                        dq = deque([s])
                        while dq:
                            u = dq.popleft()
                            if dist[u] >= 9:
                                continue
                            for v in adj[u]:
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    parent[v] = u
                                    dq.append(v)
                                elif v != parent[u] and dist[v] <= dist[u]:
                                    clen = dist[u] + dist[v] + 1
                                    if clen < 10:
                                        found_cycle = True
                                        break
                        if found_cycle:
                            break
                    
                    if not found_cycle:
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        if _verify_fast(n, edges, 4, 10):
                            return n, edges
                        break
                    
                    # Try random swaps to break cycles
                    for _ in range(10):
                        if time.time() > n_deadline:
                            break
                        # Pick random edge
                        u = random.randrange(n)
                        v = random.choice(adj[u])
                        nbrs_u = [x for x in adj[u] if x != v]
                        nbrs_v = [y for y in adj[v] if y != u]
                        if not nbrs_u or not nbrs_v:
                            continue
                        x = random.choice(nbrs_u)
                        y = random.choice(nbrs_v)
                        
                        # Check if swap is valid
                        if x == y or x in adj[v] or y in adj[u]:
                            continue
                        
                        # Perform swap
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(u)
                        adj[u].remove(x)
                        adj[y].remove(v)
                        adj[v].remove(y)
                        adj[u].append(y)
                        adj[y].append(u)
                        adj[v].append(x)
                        adj[x].append(v)
        
        # Fallback to LUW construction
        return _build_luw(k, g)
    else:
        return _build_luw(k, g)
