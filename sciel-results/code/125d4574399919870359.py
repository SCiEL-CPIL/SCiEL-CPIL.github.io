<<<<<<< SEARCH
def _greedy_walk(adj, remaining_edges, deg, start_v, rng, temperature, avoid_cycles):
    """
    Perform a single greedy walk from start_v.
    Parameters:
      temperature: controls randomness (0=deterministic greedy, 1=fully random)
      avoid_cycles: if True, avoid revisiting vertices (simple path)
    Returns (path, remaining_edges_updated, deg_updated)
    """
    path = [start_v]
    visited = {start_v} if avoid_cycles else set()
    
    while True:
        v = path[-1]
        # Find valid neighbors
        valid_neighbors = []
        for u in adj[v]:
            edge = (min(v, u), max(v, u))
            if edge in remaining_edges:
                if avoid_cycles and u in visited:
                    continue
                valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        # Choose next vertex
        if temperature <= 0:
            # Deterministic: pick neighbor with fewest remaining edges
            next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
        else:
            # Weighted random: favor low-degree neighbors
            weights = []
            for u in valid_neighbors:
                d = deg.get(u, 0)
                weights.append(1.0 / (d + 1.0))
            total_w = sum(weights)
            if total_w == 0:
                next_v = valid_neighbors[0]
            else:
                r = rng.random() * total_w
                cum = 0
                next_v = valid_neighbors[-1]
                for i, u in enumerate(valid_neighbors):
                    cum += weights[i]
                    if r <= cum:
                        next_v = u
                        break
        
        # Remove edge
        edge = (min(v, next_v), max(v, next_v))
        remaining_edges.discard(edge)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        if avoid_cycles:
            visited.add(next_v)
    
    return path
=======
def _greedy_walk(adj, remaining_edges, deg, start_v, rng, temperature, avoid_cycles):
    """
    Perform a single greedy walk from start_v.
    Parameters:
      temperature: controls randomness (0=deterministic greedy, 1=fully random)
      avoid_cycles: if True, avoid revisiting vertices (simple path)
    Returns (path, remaining_edges_updated, deg_updated)
    """
    path = [start_v]
    visited = {start_v} if avoid_cycles else set()
    
    while True:
        v = path[-1]
        # Find valid neighbors
        valid_neighbors = []
        for u in adj[v]:
            edge = (min(v, u), max(v, u))
            if edge in remaining_edges:
                if avoid_cycles and u in visited:
                    continue
                valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        # Choose next vertex
        if temperature <= 0:
            # Deterministic: pick neighbor with fewest remaining edges
            next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
        else:
            # Weighted random: favor low-degree neighbors
            weights = []
            for u in valid_neighbors:
                d = deg.get(u, 0)
                weights.append(1.0 / (d + 1.0))
            total_w = sum(weights)
            if total_w == 0:
                next_v = valid_neighbors[0]
            else:
                r = rng.random() * total_w
                cum = 0
                next_v = valid_neighbors[-1]
                for i, u in enumerate(valid_neighbors):
                    cum += weights[i]
                    if r <= cum:
                        next_v = u
                        break
        
        # Remove edge
        edge = (min(v, next_v), max(v, next_v))
        remaining_edges.discard(edge)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        if avoid_cycles:
            visited.add(next_v)
    
    return path

def _try_merge_paths(paths, adj, edge_set):
    """
    Try to merge two paths if they can be connected by an edge.
    Returns new list of paths if a merge was made, otherwise None.
    """
    n_paths = len(paths)
    if n_paths < 2:
        return None
    
    # Build edge lookup for quick checking
    # For each path, record start and end vertices
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            # Try all four combinations of connecting endpoints
            pi_start, pi_end = paths[i][0], paths[i][-1]
            pj_start, pj_end = paths[j][0], paths[j][-1]
            
            # Check if we can connect pi_end to pj_start
            if (min(pi_end, pj_start), max(pi_end, pj_start)) in edge_set:
                # Merge: reverse pj and append to pi
                merged = paths[i] + paths[j][::-1]
                new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                return new_paths
            
            # Check if we can connect pi_start to pj_start (reverse pi)
            if (min(pi_start, pj_start), max(pi_start, pj_start)) in edge_set:
                merged = paths[i][::-1] + paths[j][::-1]
                new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                return new_paths
            
            # Check if we can connect pi_end to pj_end
            if (min(pi_end, pj_end), max(pi_end, pj_end)) in edge_set:
                merged = paths[i] + paths[j]
                new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                return new_paths
            
            # Check if we can connect pi_start to pj_end
            if (min(pi_start, pj_end), max(pi_start, pj_end)) in edge_set:
                merged = paths[i][::-1] + paths[j]
                new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                return new_paths
    
    return None
>>>>>>> REPLACE

