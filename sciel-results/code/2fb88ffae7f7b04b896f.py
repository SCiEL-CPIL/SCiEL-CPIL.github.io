import time
import random
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _generate_random_regular(n, k):
    """Generate a simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Shuffle
    random.shuffle(stubs)
    
    # Pair up
    adjacency = [set() for _ in range(n)]
    valid = True
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v or v in adjacency[u]:
            valid = False
            break
        adjacency[u].add(v)
        adjacency[v].add(u)
    
    if valid:
        # Verify degrees
        for i in range(n):
            if len(adjacency[i]) != k:
                return None
        return adjacency
    return None

def _check_girth(adjacency, n, g):
    """Check if graph has girth >= g. Returns True if valid."""
    # BFS from each vertex, limit depth to g//2
    max_depth = g // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a back edge
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # For k=5, g=9, try random graphs with n < 1250
    if k == 5 and g == 9:
        # Expected girth ~ 2*ln(n)/ln(4). For n=1000, ~9.9.
        # So n=900, 1000, 1100 are good candidates.
        target_ns = [900, 950, 1000, 1050, 1100, 1150, 1200]
        
        for n in target_ns:
            if time.time() > deadline:
                break
            
            # Try a few random instances
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                adj = _generate_random_regular(n, k)
                if adj is None:
                    continue
                
                if _check_girth(adj, n, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    
    # Fallback to LUW
    return _luw_construct(k, g)
