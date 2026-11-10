import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0
    
    if k == 5 and g == 9:
        # Try LUW(5, 4) first
        n4, edges4 = _construct_luw_optimized(5, 4)
        if _check_girth(n4, edges4, 9):
            return n4, edges4
        
        # Fallback to LUW(5, 5)
        n5, edges5 = _construct_luw_optimized(5, 5)
        return n5, edges5
        
    # Fallback to original logic for other k, g
    return _construct_luw_original(k, g)

def _construct_luw_optimized(q, d):
    # Precompute rules
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # q is prime power? For q=5, it's prime.
    # We need add, sub, mul in GF(q)
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    
    n_vertices = 2 * (q ** d)
    neighbors = [[] for _ in range(n_vertices)]
    
    # Helper to convert index to (side, coeffs)
    def decode(idx):
        side = idx // (q ** d)
        val = idx % (q ** d)
        coeffs = [0] * d
        for i in range(d):
            coeffs[i] = val % q
            val //= q
        return side, coeffs
        
    # Helper to convert (side, coeffs) to index
    def encode(side, coeffs):
        val = 0
        for i in range(d):
            val = val * q + coeffs[i]
        return side * (q ** d) + val
        
    # Generate neighbors for side 0
    # Vertex (0, x) where x is tuple of d elements
    # Neighbors are (1, y)
    # y[0] is free (0..q-1)
    # y[c] = x[c] + y[a] * x[b] for side 0
    # Wait, the seed code says:
    # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
    # Note: x is the vertex on the current side.
    # For side 0, vertex is (0, x).
    # y is the vertex on side 1.
    # The relation is y[c] - x[c] = y[a] x[b] ?
    # Let's follow the seed's logic exactly.
    
    for v in range(n_vertices):
        side, x = decode(v)
        # x is list of d coeffs
        # Note: seed uses x[c] where c is 2..d
        # My x is 0..d-1
        # Seed: y = [None, first] + [0]*(d-1)
        # So y[0] is None? No, y[1] is first.
        # Let's map indices carefully.
        # Seed: x[c] for c in 2..d.
        # This implies x has indices 1..d?
        # In seed: x = (None,) + (0,)*d. So x[0] is None, x[1]..x[d] are coeffs.
        # My coeffs are x[0]..x[d-1].
        # So seed's x[c] corresponds to my coeffs[c-1].
        
        for first in range(q):
            y = [0] * (d + 1) # 1-indexed for simplicity
            y[0] = -1 # marker
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                # x[c] is my coeffs[c-1]
                # y[a] is y[a]
                # x[b] is my coeffs[b-1]
                if side == 0:
                    y[c] = add(x[c-1], mul(y[a], x[b-1]))
                else:
                    y[c] = sub(x[c-1], mul(x[a-1], y[b]))
            # Construct neighbor index
            # y[1]..y[d] are the coeffs
            neighbor_coeffs = y[1:d+1]
            neighbor_idx = encode(1 - side, neighbor_coeffs)
            
            # Avoid duplicate edges (undirected)
            # If we add edge (v, u), we don't need to add (u, v) later
            # But since we iterate all v, we will see both.
            # We can just add if v < neighbor_idx
            if v < neighbor_idx:
                neighbors[v].append(neighbor_idx)
                neighbors[neighbor_idx].append(v)
                
    edges = []
    for u in range(n_vertices):
        for v in neighbors[u]:
            if u < v:
                edges.append((u, v))
                
    return n_vertices, edges

def _check_girth(n, edges, g):
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
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    # Length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_original(k, g):
    # Copy of seed's _construct_luw
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
