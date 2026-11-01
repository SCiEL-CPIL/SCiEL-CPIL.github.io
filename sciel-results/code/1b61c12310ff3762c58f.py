import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    if k != 5:
        return _luw_construct(k, g)
        
    # Search for random 5-regular graphs with girth >= g
    # Try decreasing n from 1000 down to Moore bound
    for n in range(1000, 427, -10):
        if time.time() > deadline - 15.0:
            break
            
        # Try to find a valid graph of size n
        for attempt in range(100):
            if time.time() > deadline - 5.0:
                break
                
            edges = _generate_random_5_regular(n)
            if edges is None:
                continue
                
            if _has_girth_at_least(edges, n, g):
                return n, edges
                
    # If no smaller graph found, use L-U-W
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original L-U-W code
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

def _generate_random_5_regular(n):
    # Configuration model with retry on failures
    if n % 2 != 0 or n < 6:
        return None
    if n * 5 % 2 != 0:
        return None
        
    for _ in range(10):  # Retry a few times
        stubs = []
        for i in range(n):
            for _ in range(5):
                stubs.append(i)
        random.shuffle(stubs)
        
        adj = {i: set() for i in range(n)}
        edges = []
        valid = True
        for i in range(0, n * 5, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        if not valid:
            continue
            
        # Check degree
        if all(len(adj[i]) == 5 for i in range(n)):
            return edges
            
    return None

def _has_girth_at_least(edges, n, g):
    # Build adjacency list as lists for speed
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For girth >= g, we need to check for cycles of length 3..g-1
    # BFS from each vertex up to depth floor((g-1)/2)
    max_depth = (g - 1) // 2
    
    for start in range(n):
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
                    if dist[v] < max_depth:
                        queue.append(v)
                else:
                    # Found a cycle
                    if parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
    return True
