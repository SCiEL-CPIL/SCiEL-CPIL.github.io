<<<<<<< SEARCH
def _verify_girth_fast(n, adj, k, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g.
    adj is a list of lists of neighbors.
    """
    # Early check for regularity and simplicity
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False
        # Check for self-loops
        if i in adj[i]:
            return False

    # For each vertex, do a BFS to find the shortest cycle.
    # We stop BFS when depth reaches g. If we find a cycle of length < g, return False.
    # Optimization: We can use a single array for visited depths and a timestamp to avoid clearing.
    
    # Since n is small (320), and g is 10, a simple BFS is fast enough.
    # We need to check for cycles of length < g.
    # A cycle of length L corresponds to a path of length L-1 from a neighbor of v back to v?
    # No, standard BFS from v finds shortest paths. If we find a neighbor of v at distance d < g-1, 
    # and that neighbor is not the parent, we have a cycle of length d+1.
    
    for v in range(n):
        # dist[u] = shortest distance from v to u
        dist = [-1] * n
        dist[v] = 0
        queue = deque([v])
        
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= g - 1:
                continue
                
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = d + 1
                    queue.append(neighbor)
                else:
                    # If we see a visited node, it's part of a cycle.
                    # The length of the cycle is dist[node] + dist[neighbor] + 1?
                    # No, that's for undirected graphs if we consider the tree.
                    # Actually, if dist[neighbor] <= d, it's a back edge or cross edge.
                    # In an undirected graph BFS, if we see a visited node that is not the parent,
                    # we have a cycle. The length is dist[node] + 1 + (dist[neighbor] if neighbor is not parent? No).
                    # Let's stick to the non-backtracking walk definition or standard cycle detection.
                    
                    # Simpler: Just check if dist[neighbor] + 1 < g and neighbor is not the parent?
                    # Standard BFS doesn't track parents easily in a flat array without storing them.
                    
                    # Let's use the standard property: 
                    # If we find an edge (node, neighbor) where dist[neighbor] != -1, 
                    # the cycle length is dist[node] + dist[neighbor] + 1? 
                    # This is only true if the LCA is v. 
                    # Actually, any edge connecting two nodes at depths d1, d2 forms a cycle of length d1 + d2 + 1 
                    # only if they are in different branches. If they are in the same branch, it's shorter.
                    
                    # To be safe and correct, we use the non-backtracking walk logic from the original code 
                    # but optimized.
                    pass
        
        # The simple BFS above is tricky for exact girth without parent tracking.
        # Let's revert to the original logic but make it faster by breaking early.
        
        # Re-implementing the specific check for this function to be safe:
        visited = {}
        q = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            q.append((w, v, 1))
            
        while q:
            node, parent, depth = q.popleft()
            if depth >= g:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    q.append((neighbor, node, depth + 1))
    return True
=======
def _verify_girth_fast(n, adj, k, g):
    """
    Fast girth verification using non-backtracking walks from each vertex.
    Returns True if girth >= g.
    adj is a list of lists of neighbors.
    """
    # Early check for regularity and simplicity
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False
        if i in adj[i]:
            return False

    # Non-backtracking walk approach:
    # For each vertex v, start walks from each neighbor w of v.
    # A walk (v, w1, w2, ..., wk) where wk = v and k < g indicates a cycle of length < g.
    # We track (current_node, previous_node) to avoid backtracking.
    
    for v in range(n):
        # visited[(node, parent)] = depth (number of edges from v to node via parent)
        visited = set()
        q = deque()
        
        # Initialize: walks starting from v to each neighbor
        for w in adj[v]:
            visited.add((w, v))
            q.append((w, v, 1))
            
        while q:
            node, parent, depth = q.popleft()
            
            # If depth >= g, no need to explore further (cycles of length >= g are ok)
            if depth >= g - 1:
                continue
            
            for neighbor in adj[node]:
                # Don't backtrack
                if neighbor == parent:
                    continue
                
                # If we return to the start vertex v with a cycle shorter than g
                if neighbor == v:
                    # Cycle length is depth + 1
                    if depth + 1 < g:
                        return False
                    continue
                
                state = (neighbor, node)
                if state not in visited:
                    visited.add(state)
                    q.append((neighbor, node, depth + 1))
    
    return True
>>>>>>> REPLACE

Now let me improve the voltage graph construction with a smarter search strategy:

