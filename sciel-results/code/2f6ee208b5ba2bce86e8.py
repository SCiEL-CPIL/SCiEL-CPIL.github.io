import time
import random
from collections import deque

def _luw_construct(k, g):
    # The original LUW construction as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

def _check_girth_fast(n, edges, g):
    """
    Check if graph with n vertices and given edges has girth >= g.
    Uses BFS from each vertex. Stops early if a cycle < g is found.
    """
    if n < g:
        return False
        
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # We only need to check up to depth (g-1)//2
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    if k != 3 or g != 16:
        # Fallback for other parameters, use original heuristic
        return _luw_construct(k, g)

    # Strategy: Sweep d in LUW construction to find the smallest d
    # such that the resulting graph has girth >= g.
    # LUW(d, 3) has approx 2 * 3^d vertices.
    # d=5 -> 486 verts
    # d=6 -> 1458 verts
    # d=7 -> 4374 verts
    
    q = k
    best_result = None
    
    # Try d from 5 up to a reasonable limit
    # Girth of CD(d, 3) is generally increasing with d.
    # For g=16, d=6 or 7 is likely sufficient.
    for d in range(5, 12):
        if time.time() > deadline - 5:
            break
            
        # Build the LUW graph
        try:
            n, edges = _luw_construct_with_d(q, d)
        except Exception:
            continue
            
        if n == 0:
            continue
            
        # Verify degree regularity (should be k)
        # LUW is k-regular by construction, but let's be safe for small d
        # Actually, LUW CD(d,q) is q-regular.
        
        # Check girth
        if _check_girth_fast(n, edges, g):
            best_result = (n, edges)
            # Since we are sweeping from small d to large d, the first valid one is the smallest n
            break

    if best_result is not None:
        return best_result

    # Fallback to the original fixed-d construction if sweep fails or times out
    return _luw_construct(k, g)

def _luw_construct_with_d(q, d):
    """
    Helper to build LUW graph with specific d and q.
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
