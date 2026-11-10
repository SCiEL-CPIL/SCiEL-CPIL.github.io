import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    timeout = 110.0

    def luw_construct(k, g):
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

    def get_girth(n, edges):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = 10**9
        for i in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    def find_perfect_matching(n, adj):
        """Find a perfect matching using greedy with randomization"""
        random.seed(42)
        for attempt in range(200):
            order = list(range(n))
            random.shuffle(order)
            match = [-1] * n
            for i in order:
                if match[i] == -1:
                    nbrs = adj[i][:]
                    random.shuffle(nbrs)
                    for v in nbrs:
                        if match[v] == -1:
                            match[i] = v
                            match[v] = i
                            break
            if all(match[i] != -1 for i in range(n)):
                return match
        return None

    # Strategy: Construct (k+1)-regular graph with girth >= g+1, 
    # then remove a perfect matching to get k-regular with girth >= g
    k_plus_1 = k + 1
    g_base = g + 1
    
    # Construct (k+1)-regular graph with sufficient girth
    n_base, edges_base = luw_construct(k_plus_1, g_base)
    
    # Build adjacency for base graph
    adj = [[] for _ in range(n_base)]
    for u, v in edges_base:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find perfect matching
    match = find_perfect_matching(n_base, adj)
    
    if match is not None:
        # Create set of matching edges for quick lookup
        matching_edges = set()
        for i in range(n_base):
            j = match[i]
            if i < j:
                matching_edges.add((i, j))
        
        # Filter out matching edges
        edges_reduced = []
        for u, v in edges_base:
            if (u, v) not in matching_edges and (v, u) not in matching_edges:
                edges_reduced.append((u, v))
        
        # Verify the result
        n = n_base
        adj2 = [[] for _ in range(n)]
        for u, v in edges_reduced:
            adj2[u].append(v)
            adj2[v].append(u)
        
        # Check all vertices have degree k
        valid = True
        for i in range(n):
            if len(adj2[i]) != k:
                valid = False
                break
        
        if valid:
            # Check girth
            girth = get_girth(n, edges_reduced)
            if girth >= g:
                return n, edges_reduced
    
    # Fallback to direct LUW construction
    return luw_construct(k, g)
