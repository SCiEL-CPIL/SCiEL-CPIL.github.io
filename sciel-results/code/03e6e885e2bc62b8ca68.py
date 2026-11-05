def _check_girth(N, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length < g
    # We check for closed non-backtracking walks of length L in [4, g-2] step 2
    # If any exist, girth < g.
    
    # Optimization: Only check up to depth g//2 from each start?
    # No, a cycle of length 8 can be anywhere.
    # But if there is a cycle of length L, it appears as a closed walk.
    # We can just iterate all vertices and do a DFS to depth g//2?
    # No, that finds shortest path.
    # If we find a node at distance d < g/2 that is also reachable by another path?
    # Standard girth check:
    # For each edge (u,v), remove it, check shortest path between u and v.
    # If dist < g-1, then cycle length dist+1 < g.
    # This is O(E * (V+E)) = O(N * N) = O(N^2).
    # N=2000, N^2 = 4M. Very fast.
    
    # Let's use this method.
    # Build adjacency set for O(1) lookup? No, list is fine for BFS.
    
    # Actually, simpler: 
    # If N is large enough, girth is likely large.
    # But to be safe, let's do the edge-removal BFS check for a few edges?
    # No, must be exact.
    
    # Let's just do BFS from each node up to depth g//2.
    # If we find a back-edge to an ancestor in the BFS tree at depth > 0, 
    # the cycle length is depth + ancestor_depth + 1.
    # If this length < g, return False.
    
    for start in range(N):
        # BFS
        visited = [-1] * N
        parent = [-1] * N
        queue = [start]
        visited[start] = 0
        idx = 0
        
        while idx < len(queue):
            u = queue[idx]
            idx += 1
            depth = visited[u]
            
            if depth >= g // 2:
                continue
                
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # Cycle length is depth + visited[v] + 1
                    # But this is only valid if v is not parent[u] (handled)
                    # And we must ensure it's a simple cycle?
                    # In a tree, any cross edge creates a cycle.
                    # Length = depth(u) + depth(v) + 1
                    cycle_len = depth + visited[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True
