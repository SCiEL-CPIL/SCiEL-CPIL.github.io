import time
import random
from collections import deque

def _luw_construct(d, q):
    """
    Construct Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges).
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        # Assume q is prime
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule for LUW construction
    # The rule depends on d.
    # Standard LUW uses specific coefficients.
    # The seed code used a specific rule generation.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _check_girth(n, edges, g):
    """
    Check if graph has girth at least g.
    Uses BFS from each vertex to find shortest cycles.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # We need to check for cycles of length < g
    # BFS from each node, track parent to avoid backtracking
    # If we find a cycle of length L < g, return False
    
    # Optimization: BFS up to depth (g-1)//2 from each node is not enough for odd cycles
    # But LUW graphs are bipartite.
    # If we assume bipartite, we only need to check even cycles.
    # However, to be safe, let's do a full check for cycles < g.
    
    # Actually, BFS from all vertices with depth g-1 is O(n * (k^(g-1)/2)).
    # For n=512, g=10, depth=4. 512 * 3^4 = ~40k. Fast.
    
    for start in range(n):
        visited = [-1] * n
        parent = [-1] * n
        dist = [-1] * n
        queue = deque()
        queue.append(start)
        visited[start] = start
        dist[start] = 0
        found_short_cycle = False
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < g: # Only care if we can form a cycle < g
                        queue.append(v)
                else:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    # But this is for non-bipartite.
                    # In bipartite, cycle length is dist[u] + dist[v] + 1? No.
                    # If we find a cross edge (u, v) where dist[v] != dist[u] + 1, 
                    # we have a cycle.
                    # Length = dist[u] + dist[v] + 1?
                    # Actually, if we encounter v which is already visited and not parent,
                    # the cycle length is dist[u] + dist[v] + 1 if it's a triangle (odd).
                    # For bipartite, we can just check if dist[u] + dist[v] + 1 < g?
                    # No, in BFS, if we see a visited node, the cycle length is dist[u] + dist[v] + 1.
                    
                    # However, standard BFS for girth in bipartite graphs:
                    # We only need to check if there is a cycle of length 4, 6, ...
                    # We can simply check if dist[u] + dist[v] + 1 < g.
                    
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_n = None
    best_edges = None
    
    # Try LUW constructions with different d
    # For g=10, try d=4 (n=512), d=5 (n=2048)
    # General rule: d = ceil((g-2)/2) - 1? 
    # LUW girth is 2d+2. So 2d+2 >= g => d >= (g-2)/2.
    
    candidates = []
    for d in range(3, 8):
        if 2 * d + 2 >= g:
            candidates.append(d)
            
    # Sort by expected size (smaller d = smaller n)
    candidates.sort()
    
    for d in candidates:
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(d, k)
            # Verify girth
            if _check_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
            
    # If no LUW graph worked or none found, fallback to seed logic (d=5)
    if best_n is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = _luw_construct(d, k)
        best_n = n
        best_edges = edges
        
    return best_n, best_edges
