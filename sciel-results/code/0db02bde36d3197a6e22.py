import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # For k=3, g=16, we want to find a 3-regular graph with girth >= 16
    # and minimum number of vertices.
    # Known: smallest is 936, lower bound is 512.
    
    # Strategy: Try random regular graphs for n starting from 512 upwards.
    # For each n, generate random k-regular graphs and check girth.
    # Use non-backtracking walk check for girth verification.
    
    def check_girth(adj, n, g):
        """Check if graph has no cycle shorter than g using non-backtracking walks."""
        # For each vertex, do a BFS-like exploration with non-backtracking constraint
        # A cycle of length L < g exists iff there's a non-backtracking walk of length L
        # that returns to the start vertex.
        
        max_depth = g // 2  # For even g, we need to check up to depth g/2 - 1 for cycles
        # Actually, a cycle of length c < g means there's a path of length c-1 between two adjacent vertices
        # or a non-backtracking walk of length c returning to start.
        # Let's use the standard method: for each edge (u,v), check if there's a non-backtracking
        # path from u to v of length < g-1 that doesn't use edge (u,v) directly.
        # Simpler: BFS from each vertex, tracking parent to avoid backtracking.
        
        for start in range(n):
            # dist[v] = minimum non-backtracking distance from start to v
            # We track (node, parent) to avoid immediate backtracking
            visited = set()
            queue = deque()
            queue.append((start, -1, 0))
            visited.add((start, -1))
            
            while queue:
                node, parent, depth = queue.popleft()
                
                if depth >= g - 1:
                    continue
                
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb == start and depth + 1 < g:
                        # Found a cycle of length depth + 1 < g
                        return False
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        queue.append((nb, node, depth + 1))
        
        return True
    
    def generate_random_regular(n, k, rng):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None
        
        # Configuration model with rejection
        max_attempts = 100
        for _ in range(max_attempts):
            half_edges = n * k // 2
            # Create list of stubs
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            
            # Shuffle stubs
            rng.shuffle(stubs)
            
            # Pair up stubs to form edges
            edges = []
            edge_set = set()
            valid = True
            
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
                edges.append((u, v))
            
            if valid and len(edges) == half_edges:
                return edges
        
        return None
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    # Moore bound for (3,16) is 510, so n >= 512 (must be even for bipartite)
    min_n = 512
    max_n = 1000  # Upper bound to try
    
    rng = random.Random(42)
    
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random graphs for this n
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            edges = generate_random_regular(n, k, rng)
            if edges is None:
                continue
            
            adj = build_adj(edges, n)
            
            if check_girth(adj, n, g):
                return n, edges
    
    # Fallback: Use LUW construction if random search fails
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
