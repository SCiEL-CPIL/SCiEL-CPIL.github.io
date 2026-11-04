import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Moore bound for k=3, g=16 is 766.
    # Known best is 936.
    # We search for n < 936.
    
    def is_valid_3_regular(adj):
        # adj is list of lists
        n = len(adj)
        for i in range(n):
            if len(adj[i]) != 3:
                return False
            if len(set(adj[i])) != 3:
                return False
            for j in adj[i]:
                if i == j:
                    return False
                if i not in adj[j]:
                    return False
        return True

    def has_short_cycle(adj, max_girth):
        # Check if there is a cycle of length < max_girth
        # max_girth is g, so we check for cycles of length 3..g-1
        n = len(adj)
        # BFS from each vertex up to depth g//2
        # If we find a cycle, return True
        # To be fast, we can just check up to depth 7 for g=16
        # A cycle of length L exists if there is a path of length L-1 between two adjacent vertices
        # Or simpler: BFS from u, if we reach a neighbor of u at depth d, cycle length d+1.
        # We need to ensure no cycle length < g.
        
        # Optimized: Check girth >= g
        # This is equivalent to: for all u, the ball of radius floor((g-1)/2) is a tree.
        # For g=16, radius 7.
        radius = (g - 1) // 2
        
        for u in range(n):
            visited = {u: 0}
            queue = [(u, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist == radius:
                    continue
                for nei in adj[node]:
                    if nei in visited:
                        # Found a cycle
                        # Cycle length is dist + 1 + visited[nei]? No.
                        # If we revisit a node, it's a cycle.
                        # In a tree, we only visit each node once.
                        # If we see a node already visited, it's a cycle.
                        # The length of the cycle is not simply determined, but existence is enough.
                        return True
                    else:
                        visited[nei] = dist + 1
                        queue.append((nei, dist + 1))
        return False

    def generate_random_3_regular(n):
        # Configuration model
        # 3n half-edges
        # Pair them randomly
        # Return adjacency list or None if invalid (loops/multi-edges)
        if n < 3 or (3 * n) % 2 != 0:
            return None
        
        half_edges = list(range(3 * n))
        random.shuffle(half_edges)
        
        # Pair them
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, 3 * n, 2):
            u = half_edges[i] // 3
            v = half_edges[i+1] // 3
            
            if u == v:
                valid = False
                break
            
            # Check multi-edge
            # Since we are building, we can just add and check later
            # Or check if (u,v) already in adj[u]
            if v in adj[u]:
                valid = False
                break
            
            adj[u].append(v)
            adj[v].append(u)
            
        if not valid:
            return None
            
        # Check degrees
        for i in range(n):
            if len(adj[i]) != 3:
                return None
            if len(set(adj[i])) != 3:
                return None
                
        return adj

    # Search strategy
    # Start from 935 down to 766
    # Time budget per n?
    # Total time 110s.
    # Range is 935 - 766 + 1 = 170 values.
    # 110 / 170 = 0.65s per n.
    # In 0.65s, how many random graphs can we generate and check?
    # Generation is fast. Girth check is O(n * 3^7) ~ 2e6 ops.
    # In Python, 2e6 ops takes ~0.1-0.2s.
    # So we can do maybe 2-5 checks per n.
    # This is very few.
    # Probability of success per graph is ~ constant (e.g. 0.1?).
    # So we might fail.
    
    # Better: Start from 935, but allow more time for higher n?
    # Or just run until deadline.
    
    best_n = 936 # Fallback to known best (or LUW result)
    best_edges = None
    
    # We don't have the LUW edges easily accessible if we don't run the seed.
    # But we can run the seed logic if needed?
    # The prompt says "Keep the function names...".
    # I will implement the LUW fallback in case random search fails.
    
    # LUW implementation for fallback
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

    # Run LUW first to get a baseline valid graph
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Now try to find smaller
    # Start from 935 down
    # If n_luw < 936, start from n_luw - 1
    
    start_n = min(935, best_n - 1)
    end_n = 766
    
    for n in range(start_n, end_n - 1, -1):
        if time.time() > deadline:
            break
            
        # Try a few random graphs
        # Adjust trials based on remaining time
        remaining = deadline - time.time()
        # Estimate time per check: ~0.1s
        trials = int(remaining * 5) # 5 checks per second approx?
        trials = max(1, min(trials, 50))
        
        found = False
        for _ in range(trials):
            if time.time() > deadline:
                break
            adj = generate_random_3_regular(n)
            if adj is None:
                continue
            if not has_short_cycle(adj, g):
                # Convert adj to edges
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    return best_n, best_edges
