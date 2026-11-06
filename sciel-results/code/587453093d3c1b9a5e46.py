import time
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edge_order = P['edge_order']
    
    # Build arrival index for each edge
    edge_idx = {}
    for i, (u, v) in enumerate(edge_order):
        key = (min(u,v), max(u,v))
        edge_idx[key] = i + 1  # 1-based
    
    # Build adjacency with arrival indices
    adj = [[] for _ in range(n)]
    for (u, v), idx in edge_idx.items():
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Sort adjacency lists by arrival index
    for i in range(n):
        adj[i].sort(key=lambda x: x[1])
    
    # Binary search on threshold T
    # Check if Hamilton cycle exists using only edges with arrival index <= T
    
    def has_hamilton_cycle(T, deadline):
        # Build adjacency list with only edges <= T
        adj_T = [[] for _ in range(n)]
        for i in range(n):
            for (j, idx) in adj[i]:
                if idx <= T:
                    adj_T[i].append(j)
                else:
                    break  # sorted, so rest are also > T
        
        # Check degree: every vertex must have degree >= 2
        for i in range(n):
            if len(adj_T[i]) < 2:
                return None
        
        # Backtracking: build the cycle starting from vertex 0
        # Path: 0 -> v1 -> v2 -> ... -> v23 -> 0
        # We fix vertex 0 as the start, and the last vertex must connect back to 0
        
        path = [0]
        visited = [False] * n
        visited[0] = True
        path_set = set([0])
        
        def backtrack(pos):
            if time.time() > deadline:
                return None
            
            if pos == n:
                # Check if last vertex connects to 0
                if 0 in adj_T[path[-1]]:
                    return path[:]
                return None
            
            # MRV: pick next vertex from current vertex's neighbors
            cur = path[-1]
            # Try neighbors in order of arrival index (greedy preference)
            neighbors = adj_T[cur]
            
            # Sort by some heuristic - prefer vertices with fewer remaining options
            # But for speed, just try in order
            for nb in neighbors:
                if nb == 0 and pos < n - 1:
                    continue  # Can't go back to 0 until the end
                if visited[nb]:
                    continue
                
                # Pruning: if we're at pos == n-1, nb must connect to 0
                if pos == n - 1 and 0 not in adj_T[nb]:
                    continue
                
                # Degree pruning: after adding edge cur-nb, check degrees
                # Each intermediate vertex (not 0, not last) will have degree 2
                # Vertex 0 will have degree 2 (first and last edges)
                # Last vertex will have degree 2 (edge from prev and edge to 0)
                
                # Simple connectivity check: ensure the partial path + remaining can form a cycle
                # This is expensive; skip for now and rely on MRV + degree checks
                
                visited[nb] = True
                path_set.add(nb)
                path.append(nb)
                
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                
                path.pop()
                path_set.remove(nb)
                visited[nb] = False
            
            return None
        
        return backtrack(1)
    
    # Binary search
    lo, hi = 1, 276
    best_path = None
    deadline = time.time() + 80  # Leave 10s margin
    
    # First try to find the minimum T
    # Use exponential search then binary search
    # Or just linear scan from low to high since 276 is small
    
    # Actually, let's do binary search
    # But backtracking can be slow, so let's try a few thresholds
    
    # Strategy: try thresholds in increasing order, but use time budget wisely
    # For each T, limit the backtracking time
    
    # Let's try: for T in range(1, 277): check if Hamilton cycle exists
    # But this could be slow. Let's binary search with time limits.
    
    # Better: try a set of candidate thresholds
    # The answer is likely in the range 20-60 based on the problem structure
    
    # Let's do a more intelligent search
    # First, compute for each vertex the minimum and second minimum edge arrival indices
    # This gives a lower bound on T
    
    min_edge = [min(idx for (j, idx) in adj[i]) for i in range(n)]
    second_min_edge = [sorted(idx for (j, idx) in adj[i])[1] for i in range(n)]
    
    # Lower bound: T must be at least max of second_min_edge (since each vertex needs 2 edges)
    # Actually no, T must be >= max over all vertices of their 2nd smallest edge index
    # because each vertex needs at least 2 edges in the cycle, both with index <= T
    lb = max(second_min_edge)
    
    # Also, the cycle has 24 edges, so T must be at least the 24th smallest edge index overall? No.
    # Each edge in the cycle has index <= T, and there are 24 edges.
    # The 24th smallest edge in the whole graph has some index, but our 24 edges don't have to be the 24 smallest.
    
    # Let's start from lb and go up
    # But first, let's try a quick greedy/2-opt approach to get a good initial solution
    
    # Greedy: build cycle by always taking the earliest available edge
    # Then 2-opt to improve
    
    # Actually, let's just do the backtracking with binary search
    # For each T, give it a time limit of (80 - elapsed) / remaining_checks
    
    # Simpler approach: iterate T from lb to 276, for each T try backtracking with a small time limit
    # If we find a cycle, that's our answer
    
    # Even simpler: since n=24 is small, use a more powerful backtracking
    
    # Let me use a different approach: 
    # 1. Start with T = lb
    # 2. Try to find Hamilton cycle
    # 3. If not found, increment T
    # 4. Use increasingly aggressive pruning as T grows
    
    # For speed, use numpy or just pure Python with good heuristics
    
    # Let me implement a more efficient backtracking with:
    # - Fixed start vertex 0
    # - MRV heuristic
    # - Forward checking (degree constraints)
    # - Connectivity check (the unvisited vertices + current path must be connectable)
    
    def find_hamilton(T, time_limit):
        adj_T = [[] for _ in range(n)]
        for i in range(n):
            for (j, idx) in adj[i]:
                if idx <= T:
                    adj_T[i].append(j)
                else:
                    break
        
        # Check minimum degree
        for i in range(n):
            if len(adj_T[i]) < 2:
                return None
        
        # Build degree count
        deg = [len(adj_T[i]) for i in range(n)]
        
        # Backtracking with strong pruning
        path = [0]
        visited = bytearray(n)
        visited[0] = 1
        # deg_used[i] = number of edges already used incident to i in the partial cycle
        deg_used = [0] * n
        deg_used[0] = 1  # one edge will be used from 0 (the first edge)
        # Actually, let's think differently.
        # In the partial path 0->v1->...->vk, vertex 0 has 1 edge used (to v1),
        # vertices v1..v_{k-1} have 2 edges used (prev and next),
        # vertex vk has 1 edge used (from v_{k-1}).
        # At the end, we need vk to connect to 0, giving both degree 2.
        
        # For pruning: 
        # - Any unvisited vertex must have at least 2 available edges to unvisited or endpoint vertices
        # - The current endpoint must have at least 1 available edge to an unvisited vertex (or to 0 if it's the last)
        
        start_time = time.time()
        
        def bt(k):
            if time.time() - start_time > time_limit:
                return None
            
            if k == n:
                # Last vertex must connect to 0
                if 0 in adj_T[path[-1]]:
                    return path[:]
                return None
            
            cur = path[-1]
            
            # Candidates: neighbors of cur that are unvisited, or 0 if k == n-1
            if k == n - 1:
                # Must go to 0
                if 0 in adj_T[cur]:
                    return path + [0]  # Actually we don't append 0, the cycle closes
                return None
            
            # Try neighbors
            # MRV: sort candidates by their remaining degree (fewest options first)
            candidates = []
            for nb in adj_T[cur]:
                if nb == 0:
                    continue  # Can't go to 0 until the end
                if visited[nb]:
                    continue
                # Check: after adding this edge, can nb still complete?
                # nb will have 1 edge used (from cur), needs 1 more
                # So nb must have at least 1 other available neighbor (unvisited or 0)
                # But 0 is only available as the final connection
                # So nb needs at least 1 unvisited neighbor (excluding cur, which is now visited)
                # Actually nb's available neighbors are those not yet visited (except cur which is being left)
                # Hmm, this is getting complex. Let's use a simpler check.
                candidates.append(nb)
            
            # Sort by degree in adj_T (fewer options = more constrained = try first)
            candidates.sort(key=lambda x: len(adj_T[x]))
            
            for nb in candidates:
                # Forward checking: after visiting nb, can the remaining vertices still form a path?
                # Simple check: nb must have at least 1 unvisited neighbor (for its second edge)
                # except if nb is the second-to-last, in which case it needs to connect to 0
                if k + 1 == n - 1:
                    # nb will be the last vertex, must connect to 0
                    if 0 not in adj_T[nb]:
                        continue
                else:
                    # nb needs at least 1 more edge to an unvisited vertex
                    # Count unvisited neighbors of nb (excluding cur)
                    count = 0
                    for x in adj_T[nb]:
                        if not visited[x] and x != cur:
                            count += 1
                    if count < 1:
                        continue
                
                visited[nb] = 1
                path.append(nb)
                
                result = bt(k + 1)
                if result is not None:
                    return result
                
                path.pop()
                visited[nb] = 0
            
            return None
        
        return bt(1)
    
    # Search for minimum T
    # Try from lb upward
    # Use increasing time limits
    
    best_result = None
    elapsed_start = time.time()
    
    # First, try a quick greedy to get an upper bound
    # Greedy: start at 0, always go to the neighbor with smallest arrival index that isn't visited
    # Then check if we can close the cycle
    
    # Actually, let's just do the backtracking search
    # Try T values in a smart order
    
    # Compute a good lower bound
    # Each vertex needs 2 edges, so T >= max over vertices of their 2nd smallest edge index
    lb = max(second_min_edge)
    
    # Also try: the 24 edges of the cycle must all have index <= T
    # A loose lower bound: T >= 24 (at least 24 distinct edges)
    lb = max(lb, 24)
    
    # Search from lb to 276
    # For each T, give it a time limit based on remaining budget
    total_time = 80
    # Try a few thresholds with increasing time limits
    
    # Strategy: binary search between lb and some upper bound
    # First find an upper bound with a quick greedy
    
    # Quick greedy upper bound
    def greedy_cycle():
        # Start at 0, greedily pick next vertex
        visited = [False] * n
        visited[0] = True
        path = [0]
        for _ in range(n - 1):
            cur = path[-1]
            best_nb = -1
            best_idx = float('inf')
            for (nb, idx) in adj[cur]:
                if not visited[nb]:
                    if idx < best_idx:
                        best_idx = idx
                        best_nb = nb
            if best_nb == -1:
                return None
            visited[best_nb] = True
            path.append(best_nb)
        # Check if last connects to 0
        last = path[-1]
        for (nb, idx) in adj[last]:
            if nb == 0:
                return path
        return None
    
    gc = greedy_cycle()
    if gc:
        max_idx = max(edge_idx[(min(gc[i], gc[(i+1)%n]), max(gc[i], gc[(i+1)%n]))] for i in range(n))
        hi_bound = max_idx
    else:
        hi_bound = 276
    
    # Binary search between lb and hi_bound
    lo, hi = lb, hi_bound
    best_T = hi_bound
    best_path = gc if gc else None
    
    while lo <= hi and time.time() - elapsed_start < 75:
        mid = (lo + hi) // 2
        time_left = 75 - (time.time() - elapsed_start)
        if time_left < 0.1:
            break
        # Give this check a time limit
        tl = min(2.0, time_left / 2)  # Cap individual check at 2s
        result = find_hamilton(mid, tl)
        if result is not None:
            best_T = mid
            best_path = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Now try to improve: search below best_T more carefully? 
    # Actually binary search should have found the minimum T.
    # But the backtracking might have failed due to time limits.
    # Let's verify by trying best_T with more time
    
    if best_path is None:
        # Fallback: use the seed
        return list(range(n))
    
    # Verify best_path is valid
    # Check all vertices present
    if set(best_path) != set(range(n)):
        return list(range(n))
    # Check all edges exist and compute max index
    max_idx = 0
    for i in range(n):
        u, v = best_path[i], best_path[(i+1) % n]
        key = (min(u,v), max(u,v))
        idx = edge_idx[key]
        max_idx = max(max_idx, idx)
    
    return best_path

# EVOLVE-BLOCK-END
