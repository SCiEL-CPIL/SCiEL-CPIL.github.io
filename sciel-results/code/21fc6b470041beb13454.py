"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import numpy as np
import math
from collections import deque

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    best_n = None
    best_edges = None
    
    # Try LUW construction with various q and d
    # n = 2 * q^d
    # Target n around 243-400
    for q in [2, 3, 4, 5, 7, 8, 9, 11, 13, 16]:
        # Estimate d
        # 2 * q^d ~ 300 => q^d ~ 150
        # d ~ log(150)/log(q)
        d_min = max(3, int(math.log(100, q)))
        d_max = int(math.log(500, q)) + 2
        
        for d in range(d_min, d_max + 1):
            n = 2 * (q ** d)
            if n < 243: # Moore bound
                continue
            if best_n is not None and n >= best_n:
                continue
                
            # Construct graph
            edges = _luw_generate(q, d, k)
            
            # Verify girth
            if edges is not None and _check_girth_bfs(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback to original seed logic if LUW fails
    return _construct_fallback(k, g)

def _luw_generate(q, d, k):
    """
    Generate LUW graph.
    Vertices: (side, x) where side in {0,1}, x in {0..q-1}^d.
    Total vertices: 2 * q^d.
    """
    # The seed's rule logic:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This specific rule might be tuned for specific q?
    # The seed used XOR for q=4. For general q, we use field arithmetic.
    
    if q == 4:
        # Use the specific table from seed for q=4
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume prime field for simplicity, or just mod q if q is prime
        # For composite q like 8, 9, 16, we need field arithmetic.
        # For now, try only primes and prime powers where we can easily define fields.
        # q=2, 3, 5, 7, 11, 13 are primes.
        # q=4 is handled.
        # q=8, 9, 16: we need GF(q).
        # To keep it simple and robust, we will only use primes and q=4 in this loop.
        if q not in [2, 3, 5, 7, 11, 13]:
            return None
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1 (indices 0..d)
        # In seed: y = [None, first] + [0] * (d - 1)
        # This implies x has indices 0..d.
        # The loop in seed: for c in range(2, d + 1):
        # So x[0] is unused? Or x[0] is the "first" component?
        # Seed: start = (0, (None,) + (0,) * d)
        # So x[0] is None. x[1]..x[d] are values.
        # Let's map vertex index to (side, x_tuple).
        
        # We need to iterate over 'first' in range(q).
        # In seed: for first in range(q):
        #    y = [None, first] + [0] * (d - 1)
        #    This creates a list of length 1 + 1 + d-1 = d+1.
        #    y[0]=None, y[1]=first, y[2..d]=0.
        #    Then it updates y[2..d] based on x.
        
        # Wait, the seed's neighbour logic is specific.
        # Let's trace it.
        # vertex = (side, x) where x is tuple of length d+1.
        # x[0] is None.
        
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # Seed logic:
                # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # Build adjacency list
    n = 2 * (q ** d)
    adj = [[] for _ in range(n)]
    
    # Map (side, x_tuple) to index
    # Index = side * q^d + sum(x[i] * q^(i-1) for i in 1..d)
    # x[0] is always None.
    
    def get_index(side, x):
        # x is tuple of length d+1
        # x[0] is None
        val = 0
        for i in range(1, d + 1):
            val = val * q + x[i]
        return side * (q ** d) + val

    # Generate all vertices
    vertices = []
    for side in range(2):
        for x_tuple in itertools.product(range(q), repeat=d):
            # x_tuple corresponds to x[1]..x[d]
            full_x = (None,) + x_tuple
            idx = get_index(side, full_x)
            vertices.append((side, full_x))
            
    # For each vertex, find neighbours
    for idx, (side, x) in enumerate(vertices):
        for other_side, other_x in neighbours((side, x)):
            other_idx = get_index(other_side, other_x)
            if idx < other_idx:
                adj[idx].append(other_idx)
                adj[other_idx].append(idx)
            # Ensure no self loops or duplicates (should be simple)
            
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            # It's possible the construction doesn't yield exactly k for all params
            # or has issues.
            return None
            
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    return edges

def _check_girth_bfs(n, edges, g):
    """Check if girth >= g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        # BFS from start
        # dist[v] = distance from start
        # parent[v] = previous node in shortest path tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # But this might not be the shortest cycle through u,v
                    # Actually, BFS finds shortest paths.
                    # If we encounter a visited node v (not parent), we found a cycle.
                    # The length is dist[u] + dist[v] + 1.
                    # Is this the girth?
                    # The girth is the minimum length of any cycle.
                    # BFS from a single node finds the shortest cycle *passing through the start*?
                    # No, BFS from start finds shortest paths from start.
                    # A back edge (u, v) where u, v are already visited creates a cycle.
                    # The length of this cycle is dist[u] + dist[v] + 1.
                    # To find the global girth, we must check all such back edges for all starts?
                    # Or just one start?
                    # No, the shortest cycle in the graph might not pass through vertex 0.
                    # So we must run BFS from EVERY vertex?
                    # That is O(N * (N+E)). For N=320, E=640, this is 320 * 960 = 300k ops. Fast.
                    
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                    # Note: We don't break immediately because there might be a shorter cycle?
                    # No, dist[u] and dist[v] are fixed shortest distances.
                    # Any cycle formed by a back edge in BFS tree has length dist[u]+dist[v]+1.
                    # The minimum over all back edges in the BFS tree of a given start gives the shortest cycle passing through that start?
                    # Actually, the shortest cycle in the graph is the minimum over all vertices s of the shortest cycle passing through s.
                    # And for a fixed s, the shortest cycle passing through s is found by BFS?
                    # Yes, if we consider the BFS tree from s, any back edge (u,v) creates a cycle of length dist[u]+dist[v]+1.
                    # However, it's possible that a shorter cycle exists that doesn't use the "back edge" in the standard sense?
                    # No, any cycle in the graph will be detected as a back edge in some BFS tree.
                    # Specifically, the girth is min_{s} min_{back edge (u,v) in BFS_s} (dist_s[u] + dist_s[v] + 1).
                    
        # Optimization: If we find a cycle < g, return False.
        # We can break early if dist[u] >= g? No, dist can be large.
        
    return True

def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
