import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s
    
    # Fallback: LUW Construction
    def luw_construct(k, g):
        q = k
        # d is the dimension of the vector space
        # For girth g, we need d such that g <= 2d + 1? 
        # The LUW graph CD(d, q) has girth at least d + 5? 
        # Original code: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # For g=9, d=5. Girth >= 5+5=10? Or d+4?
        # The comment says "girth is at least d + 5 for odd d".
        # If d=5, girth >= 10. This is > 9. So it's valid.
        
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # Try to find a smaller graph using random search
    # Moore bound for odd girth g=2r+1: k + k * sum((k-1)^i for i in range(r))
    r = (g - 1) // 2
    moore_bound = k + k * sum((k - 1) ** i for i in range(r))
    
    # We want n to be even for k=5
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Try sizes from Moore bound upwards
    # Limit the search to a reasonable range, e.g., up to 1500 or until time runs out
    # The LUW graph has ~6250 vertices. Any n < 6250 is an improvement.
    
    def generate_random_regular(n, k):
        # Configuration model
        # Create list of half-edges
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)
        
        # Pair them up
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None # Loop
            if (min(u, v), max(u, v)) in edges:
                return None # Multi-edge
            edges.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)
        return list(edges), adj

    def check_girth(adj, n, g):
        # Returns True if girth >= g
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = [start]
            dist[start] = 0
            parent[start] = -2
            
            while queue:
                u = queue.pop(0)
                du = dist[u]
                if du >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # But we must be careful: if v is parent of u, it's the tree edge.
                        if v != parent[u] and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    best_n = None
    best_edges = None
    
    # First, get the LUW graph as a baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Now try to find something smaller
    # Start from Moore bound
    current_n = moore_bound
    max_n_to_try = min(1500, n_luw - 10) # Don't try to beat LUW by much if it's small, but LUW is large
    
    while current_n <= max_n_to_try:
        if time.time() > deadline:
            break
        
        # Try a few random instances of this size
        attempts = 0
        found_valid = False
        while attempts < 100:
            if time.time() > deadline:
                break
            result = generate_random_regular(current_n, k)
            if result is None:
                attempts += 1
                continue
            edges, adj = result
            if check_girth(adj, current_n, g):
                best_n = current_n
                best_edges = edges
                found_valid = True
                break
            attempts += 1
        
        if found_valid:
            # We found a valid graph of size current_n.
            # We could try smaller sizes? No, we started from Moore bound.
            # Can we try to reduce n further?
            # The loop structure allows trying smaller n if we start lower.
            # But we started at Moore bound.
            # So we can return immediately if we found one?
            # Or try to find an even smaller one?
            # Since we started at Moore bound, we can't go smaller.
            # So we can break.
            break
        
        # If not found, try slightly larger n
        current_n += 2 # Must be even

    return best_n, best_edges
