import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a smaller d that satisfies the girth
    # LUW graph CD(d, q) with q=k
    # Girth is roughly 2d+2. We need girth >= g.
    # So d >= (g-2)/2.
    
    q = k
    
    # Determine candidate d values
    # Seed logic: d = g-4 or g-5
    # New logic: Try smallest d such that 2d+2 >= g
    min_d = max(3, (g - 2 + 1) // 2) # ceil((g-2)/2)
    
    # We will try d = min_d, min_d+1, ... up to the seed's d
    seed_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    candidates = list(range(min_d, seed_d + 1))
    
    best_n = float('inf')
    best_edges = None
    
    for d in candidates:
        if time.time() > deadline:
            break
            
        # Construct LUW graph CD(d, q)
        # Reuse the neighbor logic from seed
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
        
        # Build graph
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        
        if n < best_n:
            # Check girth
            # Simple BFS girth check
            # Since it's bipartite, we can just check for cycles of length < g
            # Actually, just run BFS from each node to find shortest cycle?
            # Or just check if any cycle < g exists.
            # For bipartite, shortest cycle is even.
            # We can check if there is a cycle of length < g.
            
            # Efficient check:
            # For each edge (u, v), remove it and see if u and v are connected in the remaining graph with path length < g-2?
            # Too slow.
            
            # Just do BFS from each vertex and track distance.
            # If we visit a vertex at distance dist, and the previous distance was prev_dist,
            # cycle length = dist + prev_dist + 1.
            # But this is for undirected.
            
            # Simpler: Since N is small (1250), we can just run BFS from each vertex
            # and stop if depth reaches g-1. If we find a neighbor already visited
            # at a smaller depth, we have a cycle.
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            girth_ok = True
            for i in range(n):
                if time.time() > deadline:
                    break
                dist = [-1] * n
                dist[i] = 0
                queue = [i]
                found = False
                while queue:
                    u = queue.pop(0)
                    if dist[u] >= g - 1:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != i: # Avoid backtracking to parent
                            # Found a cycle
                            # Cycle length is dist[u] + dist[v] + 1
                            # But we need to be careful with the exact length
                            # If dist[u] + dist[v] + 1 < g, then girth < g
                            if dist[u] + dist[v] + 1 < g:
                                girth_ok = False
                                found = True
                                break
                    if found:
                        break
                if not girth_ok:
                    break
                    
            if girth_ok:
                best_n = n
                best_edges = edges
            else:
                # This d is not good enough, try larger d
                continue

    if best_edges is None:
        # Fallback to seed logic if no candidate worked (should not happen)
        # Re-run seed logic
        d = seed_d
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

    return best_n, best_edges
