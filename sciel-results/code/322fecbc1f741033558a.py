"""Construct 4-regular graph of girth >= 10."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k == 4 and g == 10:
        return _construct_4_10(deadline)
    return _luw_construct(k, g)

def _luw_construct(k, g):
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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

def _has_short_cycle(n, adj, g, max_len):
    """Check if there's a cycle of length < g using non-backtracking walks."""
    # Use BFS from each vertex to find shortest cycle
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([(start, -1)])  # (node, parent)
        
        while queue:
            u, parent_u = queue.popleft()
            for v in adj[u]:
                if v == parent_u:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append((v, u))
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return True
    return False

def _construct_4_10(deadline):
    """Try to find 4-regular graph of girth >= 10."""
    # Try L-U-W as fallback
    n_luw, edges_luw = _luw_construct(4, 10)
    
    # Try local search for better solution
    random.seed(42)
    best_n = n_luw
    best_edges = edges_luw
    
    # Try n = 320 (known best)
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        n = 320
        # Generate random 4-regular graph
        edges = _random_4_regular(n)
        
        if _has_girth_at_least(n, edges, 10):
            return n, edges
    
    return best_n, best_edges

def _random_4_regular(n):
    """Generate a random 4-regular graph on n vertices."""
    # Use configuration model
    half_edges = list(range(n)) * 4
    random.shuffle(half_edges)
    
    edges = set()
    for i in range(0, 2 * n, 2):
        u, v = half_edges[i], half_edges[i+1]
        if u != v and (v, u) not in edges:
            edges.add((u, v))
        else:
            # Retry
            for j in range(i+2, 2*n, 2):
                u2, v2 = half_edges[i], half_edges[j]
                if u2 != v2 and (v2, u2) not in edges:
                    edges.add((u2, v2))
                    half_edges[i+1], half_edges[j] = half_edges[j], half_edges[i+1]
                    break
    
    return list(edges)

def _has_girth_at_least(n, edges, g):
    """Check if graph has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
    
    # Check for short cycles using BFS
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True
