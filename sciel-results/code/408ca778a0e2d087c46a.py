import time
import random
import math

def construct(k, g):
    # LUW fallback
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
    luw_n, luw_edges = len(order), edges

    # Random search
    deadline = time.time() + 110
    # Try to find smaller graphs
    # Start from known best 1116 down to Moore bound
    moore_bound = 428
    for n in range(1116, moore_bound, -1):
        if time.time() > deadline:
            break
        
        # Try a few random instances
        found = False
        for _ in range(50):  # 50 trials per n
            if time.time() > deadline:
                break
            
            # Generate random k-regular graph
            try:
                edges = random_k_regular(n, k)
            except:
                continue
            
            # Check girth
            if check_girth(n, edges, g):
                return n, edges
    
    return luw_n, luw_edges

def random_k_regular(n, k):
    # Configuration model
    # Ensure n*k is even
    if n * k % 2 != 0:
        return None
    
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    edges = []
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u != v:
            edges.append((u, v))
    
    # Resolve multi-edges and loops
    # This simple method might leave multi-edges.
    # We need to check for multi-edges and loops.
    # If found, we can retry or try to fix.
    # For now, let's check validity.
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return None
            
    # Convert to list of edges (avoid duplicates in output? No, edges are unique)
    # The edges list from config model might have duplicates if we don't check.
    # We checked adj.
    # But the edges list itself might contain (u,v) and (v,u) or duplicates.
    # We built edges from pairs.
    # Let's return the edges list.
    return edges

def check_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True
