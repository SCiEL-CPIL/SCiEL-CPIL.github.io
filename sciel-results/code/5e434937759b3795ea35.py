import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 145
    
    if k == 4 and g == 10:
        target_n = 320
    else:
        # Moore bound
        if g % 2 == 0:
            r = g // 2
            lb = k * (k-1) ** (r-1)
        else:
            r = (g - 1) // 2
            lb = k + (k-1) * (k-1) ** (r-1)
        target_n = lb + 10

    best_n = None
    best_edges = None

    def verify_girth_np(adj, n, g):
        # adj is a numpy array of shape (n, n) with 0/1
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int8)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                # Get neighbors of u
                neighbors = np.where(adj[u] == 1)[0]
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_random_graph_np(n, k, g, seed):
        random.seed(seed)
        # Configuration model
        stubs = []
        for v in range(n):
            stubs.extend([v] * k)
        random.shuffle(stubs)
        
        adj = np.zeros((n, n), dtype=np.int8)
        edges = []
        
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                continue
            if adj[u, v]:
                continue
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append((u, v))
            
        # Check degree
        if not np.all(adj.sum(axis=1) == k):
            return None
            
        if not verify_girth_np(adj, n, g):
            return None
            
        return edges

    # Try to find a graph of size target_n
    for attempt in range(1000):
        if time.time() > deadline - 10:
            break
        edges = build_random_graph_np(target_n, k, g, seed=attempt)
        if edges is not None:
            return target_n, edges

    # Fallback to LUW
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw
