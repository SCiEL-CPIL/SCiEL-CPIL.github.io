import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # 1. Try to find a smaller graph via random search
    # Target N < 1250 (L-U-W size)
    # Moore bound is 426.
    # Let's try N from 1000 down to 500 in steps of 10.
    
    best_n = 1250
    best_edges = None
    
    # Fallback: L-U-W Construction
    # (Implemented in the else block if search fails)
    
    # Search for random 5-regular graphs
    # We need k=5.
    if k != 5:
        # If k is not 5, use L-U-W
        return _luw_construct(k, g)
        
    for n in range(1000, 427, -10):
        if time.time() > deadline - 10.0:
            break
            
        # Try to find a valid graph of size n
        for attempt in range(50): # 50 attempts per n
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
    # Configuration model
    if n % 2 != 0:
        return None
    # Each vertex has 5 stubs
    stubs = []
    for i in range(n):
        for _ in range(5):
            stubs.append(i)
    random.shuffle(stubs)
    
    adj = {i: set() for i in range(n)}
    edges = []
    for i in range(0, n * 5, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if v in adj[u]:
            return None # Multi-edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 5:
            return None
            
    return edges

def _has_girth_at_least(edges, n, g):
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length < g
    # g=9, so check cycles 1..8
    # 1: loops (handled)
    # 2: multi-edges (handled)
    # 3..8: use BFS
    
    for start in range(n):
        # BFS from start
        dist = {start: 0}
        parent = {start: -1}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= 4: # Max depth 4 for girth 9? 
                # If we find a neighbor at depth d, and it is visited at depth d', 
                # cycle length is d + d' + 1.
                # We want to avoid cycles < 9.
                # If d=4, d'=4, cycle=9. OK.
                # If d=4, d'=3, cycle=8. Bad.
                # So we need to check if any edge connects two nodes with dist sum + 1 < 9.
                pass
                
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    # Only if v is not parent of u
                    if parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
    return True
