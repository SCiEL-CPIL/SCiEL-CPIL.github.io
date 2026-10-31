import time
import random
from collections import deque

def construct(k, g):
    if k != 3:
        return _fallback(k, g)
        
    # Target girth 16
    # Base graph K3,3
    # Vertices 0,1,2 and 3,4,5
    # Edges:
    base_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    edge_idx = {e: i for i, e in enumerate(base_edges)}
    
    deadline = time.time() + 115
    
    # Moore bound for n(3,16) is 768.
    # n = 6m. m >= 128.
    m = 128
    
    while time.time() < deadline:
        # Try random voltage assignments for this m
        # Number of tries per m
        # If m is large, probability of failure is low.
        # If m is small, probability is high.
        max_tries = 2000 if m < 150 else 500
        
        for _ in range(max_tries):
            if time.time() > deadline - 5:
                break
                
            # Random voltages
            voltages = [random.randint(0, m-1) for _ in range(9)]
            
            # Build graph
            # Vertices are (v, i) -> v*m + i
            n = 6 * m
            adj = [[] for _ in range(n)]
            
            for i in range(m):
                for u, v in base_edges:
                    e_idx = edge_idx[(u,v)]
                    c = voltages[e_idx]
                    # u -> v
                    u_node = u * m + i
                    v_node = v * m + (i + c) % m
                    adj[u_node].append(v_node)
                    # v -> u
                    # voltage is -c
                    # v_node -> u_node + (-c)
                    # u_node_next = (u, (i+c-c) % m) = (u, i)
                    # So adj[v_node].append(u_node)
                    adj[v_node].append(u_node)
            
            # Check girth
            if _check_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
        m += 1

    # Fallback if no good m found
    # Just return a large random regular graph?
    # Or the LUW one?
    # Let's try a large m one last time
    return _luw_construct(k, g)

def _check_girth(adj, n, g):
    # Check if girth >= g
    # For bipartite graphs, girth is even.
    # We check if there is a cycle of length < g.
    # BFS from each node up to depth (g-1)//2
    # If we find a back edge, cycle length = dist[u] + dist[v] + 1?
    # In BFS tree, if we see edge (u,v) and v is visited and v is not parent[u],
    # then we have a cycle.
    # The length of the cycle is dist[u] + dist[v] + 1?
    # Only if v is an ancestor.
    # If v is not an ancestor, it's a cross edge.
    # But in a graph with high girth, the BFS ball is a tree.
    # So if we see ANY visited node other than parent, it implies a cycle.
    # The length of the shortest cycle through u will be found.
    
    max_depth = (g - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # v is visited
                    if v != parent[u]:
                        # Found a cycle
                        # Cycle length is dist[u] + dist[v] + 1?
                        # This formula works if v is an ancestor.
                        # If v is in a different branch, the cycle is longer or complex.
                        # However, if girth > 2*max_depth, no such edge should exist.
                        # If it exists, girth < g.
                        return False
    return True

def _luw_construct(k, g):
    # Keep the old fallback
    d, q = _luw_params(k, g)
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

def _luw_params(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return d, q

def _fallback(k, g):
    return _luw_construct(k, g)
