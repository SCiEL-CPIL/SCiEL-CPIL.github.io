def construct(k, g):
    import time
    import random
    from collections import deque
    
    start_time = time.time()
    deadline = start_time + 115
    
    def moore_bound(k, g):
        # For odd girth g = 2r + 1
        # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
        r = (g - 1) // 2
        n = 1
        term = 1
        for _ in range(r):
            n += k * term
            term *= (k - 1)
        return n

    n_min = moore_bound(k, g)
    
    # We will try to construct a graph with n vertices
    # Start from n_min and go up
    # The LUW construction in the seed likely produced a large n.
    # We hope to find a smaller one.
    
    def try_build(n, k, g, max_attempts=50):
        """
        Try to build a k-regular graph on n vertices with girth >= g
        using randomized greedy edge addition.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            adj = [set() for _ in range(n)]
            deg = [0] * n
            edges = []
            
            # Helper: check if adding (u, v) creates a cycle of length < g
            # This is equivalent to checking if there is a path of length < g-1 between u and v
            # Note: if u and v are adjacent, path length 1 exists.
            # We want to avoid cycles of length L < g.
            # Adding edge (u,v) creates a cycle of length L+1 if there is a path of length L.
            # So we need NO path of length 1, 2, ..., g-2.
            # i.e., dist(u, v) must be >= g-1 or unreachable (but graph will be connected eventually).
            # Actually, if dist(u,v) < g-1, adding (u,v) creates a cycle of length dist(u,v)+1 < g.
            # So we require dist(u, v) >= g-1.
            
            def has_short_path(u, v, limit):
                # BFS from u to v, stop if depth > limit
                if u == v:
                    return True # Loop, not allowed
                if v in adj[u]:
                    return True # Path length 1
                visited = {u: 0}
                dq = deque([u])
                while dq:
                    curr = dq.popleft()
                    depth = visited[curr]
                    if depth >= limit:
                        continue
                    for nei in adj[curr]:
                        if nei == v:
                            return True # Found path of length depth+1 <= limit
                        if nei not in visited:
                            visited[nei] = depth + 1
                            dq.append(nei)
                return False

            # Greedy construction
            # We need to add edges until all degrees are k.
            # To avoid getting stuck, we prioritize vertices with low degree.
            
            success = True
            while True:
                if time.time() > deadline:
                    return None
                
                # Find vertices with degree < k
                candidates = [i for i in range(n) if deg[i] < k]
                if not candidates:
                    break # Done
                
                # Sort by degree to fill up low degree vertices first
                candidates.sort(key=lambda x: deg[x])
                
                edge_added = False
                # Try to find an edge for the lowest degree vertex
                u = candidates[0]
                
                # Potential partners: vertices with degree < k, not u, not adjacent
                potential_v = [v for v in range(n) if v != u and deg[v] < k and v not in adj[u]]
                
                if not potential_v:
                    # Stuck: u cannot connect to any other vertex with capacity
                    # This means we failed this attempt
                    success = False
                    break
                
                random.shuffle(potential_v)
                
                for v in potential_v:
                    # Check if adding (u, v) is safe
                    # We need dist(u, v) >= g-1
                    if has_short_path(u, v, g - 2):
                        continue
                    
                    # Safe to add
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.append((u, v))
                    edge_added = True
                    break
                
                if not edge_added:
                    # No valid edge found for u in this pass
                    # Try next vertex? Or fail?
                    # If we can't connect u to anyone, we fail.
                    # But maybe u can connect to a vertex that is currently full? No, degree must be exactly k.
                    # So if no potential_v works, we fail.
                    success = False
                    break
            
            if success:
                # Verify regularity
                if all(d == k for d in deg):
                    # Verify girth exactly
                    if verify_girth(n, edges, g):
                        return n, edges
        return None

    def verify_girth(n, edges, g):
        """Exact check: no cycle of length < g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        # To speed up, we can use a single BFS with parent tracking for non-backtracking?
        # Standard BFS detects shortest path.
        # If shortest path from u to v is d, and (u,v) is an edge, cycle length is d+1.
        # We need to check all edges (u,v) if dist(u,v) < g-1.
        # But dist(u,v) in the graph WITHOUT the edge (u,v) is what matters.
        # Actually, if we have the full graph, and we find a cycle of length L < g, we fail.
        # A cycle of length L exists if there is a path of length L-1 between two adjacent vertices.
        
        # Efficient check:
        # For each vertex s, run BFS.
        # Record dist[s][v] and parent[s][v].
        # For each edge (u, v), if dist[u][v] < g-1, then there is a cycle of length dist[u][v]+1.
        # Wait, dist[u][v] is the shortest path in the full graph.
        # If the shortest path uses the edge (u,v), it's length 1.
        # We need to check if there is a path of length L < g-1 that does NOT use (u,v).
        # This is equivalent to checking if the second shortest path or any path < g-1 exists.
        
        # Simpler: Just BFS from each node, and check for any back edge that closes a cycle < g.
        # Standard BFS tree:
        # For each node u, for each neighbor v:
        #   if v is parent of u, skip.
        #   if dist[u] + dist[v] + 1 < g:
        #       return False (cycle found)
        #   if dist[u] + 1 < dist[v]: update
        # This works if we do BFS from each start node.
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        dq.append(v)
                    else:
                        # If v is not parent of u, and u is not parent of v
                        # And dist[u] + dist[v] + 1 < g
                        if v != parent[u] and u != parent[v]:
                            if dist[u] + dist[v] + 1 < g:
                                return False
        return True

    best_result = None
    
    # First, try the algebraic LUW construction which is deterministic and likely valid
    # for appropriate parameters. This avoids the randomness and potential failure of greedy.
    def build_luw(q, d):
        # Simplified LUW from seed
        if d < 2:
            return 0, []
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
                        y[c] = (x[c] + y[a] * x[b]) % q
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q
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

    # Try LUW with parameters that might yield n >= Moore bound and girth >= g
    # For k=5, d=4 gives n=1250. d=3 gives n=250 (too small, < 426).
    # So we start from d=4.
    for d in range(4, 8):
        if time.time() > deadline - 10:
            break
        try:
            n_luw, edges_luw = build_luw(k, d)
            if n_luw < n_min:
                continue
            if is_regular_simple(n_luw, edges_luw, k) and verify_girth(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue

    # If LUW failed, fall back to randomized greedy search
    current_n = n_min
    # Extend the search range to ensure we find a solution
    max_n_to_try = n_min + 1000 
    
    while current_n <= max_n_to_try and time.time() < deadline:
        if time.time() > deadline - 10:
            break
            
        # Use fewer attempts for larger n to save time, as larger n is easier
        attempts = 50 if current_n < n_min + 200 else 20
        result = try_build(current_n, k, g, max_attempts=attempts)
        if result:
            n, edges = result
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
            return n, edges
        
        # Increase n
        # Parity constraints? k*n must be even.
        # If k is odd, n must be even.
        if k % 2 == 1:
            if current_n % 2 != 0:
                current_n += 1
            else:
                current_n += 2
        else:
            current_n += 1

    if best_result:
        return best_result

    # If all else fails, return the smallest valid LUW graph we could potentially generate
    # or a known valid graph if we had one.
    # Since we must return a valid graph, and we might have failed all,
    # we try one last LUW with a larger d to ensure validity, even if n is large.
    for d in range(8, 10):
        try:
            n_luw, edges_luw = build_luw(k, d)
            if is_regular_simple(n_luw, edges_luw, k) and verify_girth(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            pass

    # This should not be reached if LUW works.
    # Return a dummy to indicate failure, but this scores 0.
    return (0, [])

def is_regular_simple(n, edges, k):
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return False
        seen.add(key)
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    return True
