<<<<<<< SEARCH
def _verify_girth_np(adj, n, g):
    """Vectorized girth check using non-backtracking BFS from all vertices simultaneously.
    Returns True iff girth >= g."""
    if n == 0:
        return True
    # adj: list of lists, degree k=3
    # Build edge arrays for fast lookup
    # For each vertex, store its 3 neighbors
    k = 3
    nbr = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        nbr[i] = adj[i]
    
    # State: (current_vertex, prev_vertex) -> encoded as vertex * n + prev
    # We do BFS from all starts simultaneously using a frontier of states
    # But that's O(n * states). Instead, do it per-start but with numpy for the frontier.
    
    # Actually, for n up to ~1000 and g=16, non-backtracking BFS from each vertex
    # visits at most 3*2^(g-2) states. Let's do per-start but with numpy arrays.
    
    max_depth = g - 1  # we need to check cycles of length < g, i.e., up to g-1
    
    for start in range(n):
        # frontier: list of (v, prev) states at current depth
        # Use arrays for speed
        frontier_v = np.array([start], dtype=np.int32)
        frontier_p = np.array([-1], dtype=np.int32)
        visited = np.full(n * n, -1, dtype=np.int32)
        visited[start * n + (-1 % n)] = 0  # mark start state
        
        depth = 0
        while len(frontier_v) > 0 and depth < max_depth:
            depth += 1
            # For each state (v, prev), expand to neighbors of v except prev
            # Get neighbors of all frontier vertices
            all_nbrs = nbr[frontier_v]  # shape (len, 3)
            
            # Filter out prev
            valid = np.ones((len(frontier_v), 3), dtype=bool)
            valid[:, :] = all_nbrs != frontier_p[:, None]
            
            # Flatten valid transitions
            fv = all_nbrs[valid]
            fp = frontier_v[valid]  # the current vertex becomes the prev for next state
            
            # Check if any transition goes back to start (cycle found)
            if depth >= 3:
                if np.any(fv == start):
                    return False
            
            # Filter already visited
            state_ids = fv * n + fp
            mask = visited[state_ids] == -1
            if not np.any(mask):
                break
            fv = fv[mask]
            fp = fp[mask]
            state_ids = state_ids[mask]
            
            visited[state_ids] = depth
            frontier_v = fv
            frontier_p = fp
    
    return True
=======
def _verify_girth_np(adj, n, g):
    """Vectorized girth check using non-backtracking BFS from all vertices simultaneously.
    Returns True iff girth >= g."""
    if n == 0:
        return True
    # adj: list of lists, degree k=3
    # Build edge arrays for fast lookup
    # For each vertex, store its 3 neighbors
    k = 3
    nbr = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        nbr[i] = adj[i]
    
    max_depth = g - 1  # we need to check cycles of length < g, i.e., up to g-1
    
    for start in range(n):
        # frontier: list of (v, prev) states at current depth
        # Use arrays for speed
        frontier_v = np.array([start], dtype=np.int32)
        frontier_p = np.array([-1], dtype=np.int32)
        visited = np.full(n * n, -1, dtype=np.int32)
        # Do not mark start state with sentinel, handle depth 1 naturally
        
        depth = 0
        while len(frontier_v) > 0 and depth < max_depth:
            depth += 1
            # For each state (v, prev), expand to neighbors of v except prev
            # Get neighbors of all frontier vertices
            all_nbrs = nbr[frontier_v]  # shape (len, 3)
            
            # Filter out prev
            # frontier_p is 1D (N,), need to broadcast to (N, 3) for comparison with all_nbrs (N, 3)
            # Use -1 as sentinel for prev, which is not a valid vertex index, so all_nbrs != -1 is True
            valid = all_nbrs != frontier_p[:, None]
            
            # Flatten valid transitions
            fv = all_nbrs[valid]
            # The new prev is the current vertex v. 
            # frontier_v is (N,), valid is (N, 3). 
            # We need to broadcast frontier_v to (N, 3) and index with valid.
            fp = frontier_v[:, None][valid]
            
            # Check if any transition goes back to start (cycle found)
            # A cycle of length L is found if we return to start at depth L.
            # Since we start at depth 0, returning at depth d means cycle length d.
            # We want girth >= g, so we fail if we find a cycle of length < g.
            # The BFS finds shortest paths. If we hit start, the length is `depth`.
            if depth >= 3: # girth 3 is impossible for 3-regular simple graph? No, triangles exist.
                if np.any(fv == start):
                    return False
            
            # Filter already visited
            # state is (fv, fp). Encode as fv * n + fp.
            # Note: fp can be -1? No, fp comes from frontier_v which are valid vertices.
            # So fp is in 0..n-1.
            state_ids = fv * n + fp
            mask = visited[state_ids] == -1
            if not np.any(mask):
                break
            fv = fv[mask]
            fp = fp[mask]
            state_ids = state_ids[mask]
            
            visited[state_ids] = depth
            frontier_v = fv
            frontier_p = fp
    
    return True
>>>>>>> REPLACE

