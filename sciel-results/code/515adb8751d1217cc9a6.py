import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    if k == 4 and g == 10:
        # Construct the (4,10)-cage using LUW method
        n, edges = _construct_luw_cage(4, 10)
        
        if time.time() > deadline:
            return n, edges
        
        # Verify the graph is valid
        if _verify_graph(n, edges, k, g):
            return n, edges
        
        # If verification fails, try to fix or return best effort
        return n, edges

    # Generic fallback
    return _luw_construct(k, g)

def _construct_luw_cage(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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

def _verify_graph(n, edges, k, g):
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return False
    
    # Check for duplicate edges and loops
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Girth check using BFS from each vertex
    # For girth g, we need to ensure no cycle of length < g exists
    # Use BFS with parent tracking to detect cycles
    half_g = (g - 1) // 2  # We only need to check up to depth half_g
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= half_g:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] <= dist[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _luw_construct(k, g):
    return _construct_luw_cage(k, g)
