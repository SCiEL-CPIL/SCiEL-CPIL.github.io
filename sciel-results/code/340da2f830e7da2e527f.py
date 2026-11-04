import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using tree embedding.
    """
    start_time = time.time()
    time_limit = 115.0
    
    # For girth g, BFS from any vertex to depth floor((g-1)/2) must be a tree
    # For odd g=9: depth 4, tree has 1 + k + k^2 + ... + k^4 nodes
    # For even g=8: depth 3, tree has 1 + k + k^2 + ... + k^3 nodes
    
    if g % 2 == 1:
        # Odd girth: center vertex, depth = (g-1)//2
        depth = (g - 1) // 2
    else:
        # Even girth: edge center, depth = g//2 - 1
        depth = g // 2 - 1
    
    # Number of vertices in the tree
    tree_size = sum(k**i for i in range(depth + 1))
    
    # We need n >= tree_size, and n*k must be even
    # Also, the number of "boundary" vertices (at max depth) must be even or we can handle it
    boundary_count = k**depth
    
    # Start from a reasonable n
    # Moore bound for odd girth: n >= 1 + k + k^2 + ... + k^depth
    # For even girth: n >= 2*(1 + k + ... + k^(depth-1))
    
    # Try different n values
    best_n = None
    best_edges = None
    
    # Helper: check if graph has girth >= g using non-backtracking walks
    def check_girth(adj, n, g):
        """Returns True if girth >= g, False otherwise."""
        # Use BFS from each vertex, tracking parent to avoid backtracking
        for start in range(n):
            # dist[v] = distance from start, parent[v] = previous vertex
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in range(n):
                    if adj[u, v] and v != parent[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Found a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # But we need to be careful: this might not be the shortest
                            # For our purposes, if any cycle < g, return False
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True
    
    # Faster check using numpy
    def check_girth_fast(adj, n, g):
        """Check girth >= g. Returns True if valid."""
        adj_list = [np.where(adj[i])[0] for i in range(n)]
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adj_list[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    # Build a BFS tree and add edges
    def try_construct(n, k, g, attempts=50):
        """Try to construct a valid graph with n vertices."""
        if (n * k) % 2 != 0:
            return None
        
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                return None
            
            # Step 1: Create a random tree structure
            # We'll create a random k-ary tree of the required depth
            # Then add edges to make it k-regular
            
            # Initialize adjacency
            adj = np.zeros((n, n), dtype=bool)
            
            # Create a random spanning tree first
            # Actually, let's use a different approach:
            # Start with a random k-regular graph, then try to increase girth
            
            # Random k-regular graph using configuration model
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            
            for _ in range(20):  # Try different random pairings
                random.shuffle(stubs)
                
                adj = np.zeros((n, n), dtype=bool)
                valid = True
                
                for i in range(n * k // 2):
                    u = stubs[2 * i]
                    v = stubs[2 * i + 1]
                    if u == v or adj[u, v]:
                        valid = False
                        break
                    adj[u, v] = True
                    adj[v, u] = True
                
                if not valid:
                    continue
                
                # Check degrees
                degrees = adj.sum(axis=1)
                if not np.all(degrees == k):
                    continue
                
                # Check girth
                if check_girth_fast(adj, n, g):
                    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                    return n, edges
            
        return None
    
    # Try increasing n from a lower bound
    # For k=5, g=9: lower bound is around 428
    # Let's try from 430 to 1200
    
    if g % 2 == 1:
        # Odd girth
        moore = sum(k**i for i in range(depth + 1))
    else:
        # Even girth
        moore = 2 * sum(k**i for i in range(depth))
    
    # Ensure moore is even if needed
    if (moore * k) % 2 != 0:
        moore += 1
    
    # Try n values
    for n in range(moore, min(moore + 800, 1300), 2):
        if time.time() - start_time > time_limit:
            break
        
        result = try_construct(n, k, g, attempts=20)
        if result is not None:
            best_n, best_edges = result
            break
    
    # Fallback if nothing found
    if best_n is None:
        # Use a simple construction: cycle of cycles
        # This won't be optimal but will be valid
        # For k=5, we can create a 5-regular graph by taking many 5-cycles
        # and connecting them in a tree-like fashion
        
        # Simple fallback: use the seed-like approach
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
        best_n = len(order)
        best_edges = edges
    
    return best_n, best_edges
