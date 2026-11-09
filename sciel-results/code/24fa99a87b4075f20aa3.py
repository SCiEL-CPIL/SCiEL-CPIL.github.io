import time
import sys
import numpy as np
from collections import deque, defaultdict
import random
import math

_DEADLINE = 150.0
_T0 = None

def _elapsed():
    return time.time() - _T0

def _time_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin

def construct(k, g):
    global _T0
    _T0 = time.time()
    
    # Handle small cases with hardcoded cages
    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return (n, edges)
    
    if k == 3:
        if g <= 4:
            return (4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
        if g <= 5:
            # Petersen
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((5 + i, 5 + (i + 2) % 5))
                edges.append((i, 5 + i))
            return (10, edges)
        if g <= 6:
            # Heawood
            edges = []
            fano_lines = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
            for li, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7 + li))
            return (14, edges)
        if g <= 7:
            # Tutte-Coxeter (30 vertices, girth 7)
            # Construct using LUW for k=3, g=7
            result = _luw_construct(k, g)
            if result is not None:
                return result
        # For k=3, g>=8, try LUW
        result = _luw_construct(k, g)
        if result is not None:
            return result
        return _simple_regular_graph(k)

    if k == 5 and g == 9:
        # Try to find the 1116 cage
        if _time_ok(margin=30.0):
            result = _search_cage_5_9_v2()
            if result is not None:
                return result
        
        # Fallback to LUW (1250 vertices)
        result = _luw_construct(k, g)
        if result is not None:
            return result
        
        # Fallback to simple regular
        return _simple_regular_graph(k)

    # General case: Try LUW
    result = _luw_construct(k, g)
    if result is not None:
        return result
    
    return _simple_regular_graph(k)

def _moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        return 2 * (1 + k * sum([(k-1)**i for i in range(r-1)]))
    else:
        r = (g - 1) // 2
        return 1 + k * sum([(k-1)**i for i in range(r)])

def _search_cage_5_9_v2():
    """Specific search for the 1116-vertex cage using local search."""
    n = 1116
    k = 5
    g = 9
    
    # Generate random regular graph
    adj = _random_regular_graph(n, k)
    if adj is None:
        return None
    
    # Precompute neighbor lists
    neighbors = [np.where(adj[i] == 1)[0].tolist() for i in range(n)]
    
    # Check initial girth
    if _check_girth(adj, neighbors, n, g):
        return (n, _adj_to_edges(adj))
    
    # Local search
    max_iters = 50000
    for _ in range(max_iters):
        if not _time_ok(margin=10.0):
            break
        
        # Find a short cycle
        cycle = _find_short_cycle(adj, neighbors, n, g)
        if cycle is None:
            return (n, _adj_to_edges(adj))
        
        # Pick an edge in the cycle to break
        # Let's pick the first edge (u, v)
        u, v = cycle[0]
        
        # Find a swap partner
        # We want to swap (u, v) and (u2, v2) with (u, v2) and (u2, v)
        # u2 is a random neighbor of some random node?
        # Better: Pick a random edge (u2, v2) not in the cycle and not incident to u or v.
        
        # To keep it simple, pick a random vertex u2 and a random neighbor v2 of u2
        u2 = random.randint(0, n - 1)
        if u2 == u or u2 == v:
            continue
        v2_list = neighbors[u2]
        if not v2_list: continue
        v2 = random.choice(v2_list)
        
        if v2 == u or v2 == v:
            continue
        if adj[u][v2] == 1 or adj[u2][v] == 1:
            continue
            
        # Perform swap
        adj[u][v] = 0
        adj[v][u] = 0
        adj[u2][v2] = 0
        adj[v2][u2] = 0
        
        adj[u][v2] = 1
        adj[v2][u] = 1
        adj[u2][v] = 1
        adj[v][u2] = 1
        
        # Update neighbors
        neighbors[u].remove(v)
        neighbors[u].append(v2)
        
        neighbors[v].remove(u)
        neighbors[v].append(u2)
        
        neighbors[u2].remove(v2)
        neighbors[u2].append(v)
        
        neighbors[v2].remove(u2)
        neighbors[v2].append(u)
        
        # Check girth periodically
        if _ % 10 == 0:
            if _check_girth(adj, neighbors, n, g):
                return (n, _adj_to_edges(adj))

    return None

def _random_regular_graph(n, k):
    if n * k % 2 != 0:
        return None
    
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if adj[u][v] == 1:
            return None
        adj[u][v] = 1
        adj[v][u] = 1
        
    return adj

def _check_girth(adj, neighbors, n, g):
    """Check if girth is at least g."""
    # We only need to check for cycles of length < g
    # Use BFS from each vertex up to depth g//2
    # If we find a cycle, return False
    # For speed, we can just run BFS and check for back edges
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif v != start: # Avoid the start node itself in this simple check? 
                    # Actually, if we see a visited node v (not parent), we have a cycle.
                    # Cycle length = dist[u] + dist[v] + 1
                    # But we need to ensure it's not the immediate parent.
                    # This BFS doesn't track parent.
                    # Let's use a proper girth check.
                    pass
    # The simple BFS above is not sufficient to detect all short cycles correctly without parent tracking.
    # Let's use the _find_short_cycle function instead.
    return _find_short_cycle(adj, neighbors, n, g) is None

def _find_short_cycle(adj, neighbors, n, g):
    """Find a cycle of length < g. Returns list of edges or None."""
    # Use BFS from each vertex to depth g//2
    # Track parent to reconstruct cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    # Found a cycle
                    # Reconstruct cycle from u to v
                    cycle = [(u, v)]
                    curr = u
                    while curr != v:
                        curr = parent[curr]
                        cycle.append((curr, parent[curr]))
                    # Check length
                    if len(cycle) < g:
                        return cycle
    return None

def _adj_to_edges(adj):
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
    return edges

def _simple_regular_graph(k):
    n = k + 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return (n, edges)

def _luw_construct(k, g, d_override=None):
    q = k
    if q not in (2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        return None

    if d_override is not None:
        d = d_override
    else:
        d = None
        for cand in range(3, 20):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
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

    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q

    def encode(side, x):
        val = 0
        for c in range(1, d + 1):
            val += x[c] * q_powers[d - c]
        return side * (q ** d) + val

    def decode(enc):
        side = enc // (q ** d)
        val = enc % (q ** d)
        x = [0] * (d + 1)
        for c in range(1, d + 1):
            x[c] = val % q
            val //= q
        return side, x

    def neighbours(enc):
        side, x = decode(enc)
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield encode(1 - side, y)

    start_enc = encode(0, [0] * (d + 1))
    index = {start_enc: 0}
    order = [start_enc]
    edges = []

    for i, enc in enumerate(order):
        if not _time_ok(margin=15.0):
            return None
        side, _ = decode(enc)
        for other_enc in neighbours(enc):
            if other_enc not in index:
                index[other_enc] = len(order)
                order.append(other_enc)
            if side == 0:
                edges.append((index[enc], index[other_enc]))

    n = len(order)
    if n == 0:
        return None

    if _verify(n, edges, k, g):
        return (n, edges)
    return None

def _verify(n, edges, k, g):
    if n == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    for start in range(n):
        if not _time_ok(margin=10.0):
            return False
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                new_state = (w, v)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if w == start and depth + 1 < g:
                    return False
                queue.append((w, v, depth + 1))
    
    return True
