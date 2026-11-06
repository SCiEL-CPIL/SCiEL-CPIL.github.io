<<<<<<< SEARCH
def _beam_search_longest_path(n, remaining_edges, start_node, beam_width=20, max_depth=100):
    """
    Finds a long simple path starting from start_node using beam search.
    Returns the path as a list of vertices.
    """
    if not remaining_edges:
        return []
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    if start_node not in rem_adj:
        return []

    # Beam: list of (path_tuple, score)
    # Score is length
    beam = [((start_node,), 1)]
    
    for _ in range(max_depth):
        next_beam = []
        for path_tuple, score in beam:
            current = path_tuple[-1]
            visited = set(path_tuple)
            
            neighbors = [nb for nb in rem_adj[current] if nb not in visited]
            
            if not neighbors:
                # Dead end, keep this path as a candidate
                next_beam.append((path_tuple, score))
                continue
            
            # Generate extensions
            for nb in neighbors:
                new_path = path_tuple + (nb,)
                # Heuristic: prefer nodes with higher remaining degree?
                # Or just length.
                new_score = len(new_path)
                next_beam.append((new_path, new_score))
                
        if not next_beam:
            break
            
        # Sort by score descending and keep top beam_width
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]
        
        # Check if any path can be extended
        can_extend = False
        for path_tuple, _ in beam:
            current = path_tuple[-1]
            visited = set(path_tuple)
            if any(nb not in visited for nb in rem_adj[current]):
                can_extend = True
                break
        if not can_extend:
            break
            
    if not beam:
        return []
        
    # Return the longest path
    best_path = max(beam, key=lambda x: len(x[0]))[0]
    return list(best_path)
=======
def _beam_search_longest_path(n, remaining_edges, start_node, beam_width=30, max_depth=150):
    """
    Finds a long simple path starting from start_node using beam search.
    Uses a composite score that considers both path length and connectivity.
    Returns the path as a list of vertices.
    """
    if not remaining_edges:
        return []
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    if start_node not in rem_adj:
        return []

    # Beam: list of (path_tuple, score)
    # Score combines length and connectivity heuristic
    def compute_score(path_tuple):
        length = len(path_tuple)
        # Bonus for paths that end at high-degree nodes (more potential to extend)
        end_node = path_tuple[-1]
        end_deg = len(rem_adj[end_node])
        # Penalize if the path has created isolated edges
        isolated_penalty = 0
        visited = set(path_tuple)
        for u, v in remaining_edges:
            if u in visited and v not in visited:
                # Edge from visited to unvisited - this is good
                pass
            elif u not in visited and v in visited:
                pass
        return length * 10 + end_deg * 2
    
    beam = [((start_node,), compute_score((start_node,)))]
    
    for _ in range(max_depth):
        next_beam = []
        for path_tuple, score in beam:
            current = path_tuple[-1]
            visited = set(path_tuple)
            
            neighbors = [nb for nb in rem_adj[current] if nb not in visited]
            
            if not neighbors:
                # Dead end, keep this path as a candidate
                next_beam.append((path_tuple, score))
                continue
            
            # Generate extensions
            for nb in neighbors:
                new_path = path_tuple + (nb,)
                new_score = compute_score(new_path)
                next_beam.append((new_path, new_score))
                
        if not next_beam:
            break
            
        # Sort by score descending and keep top beam_width
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]
        
        # Check if any path can be extended
        can_extend = False
        for path_tuple, _ in beam:
            current = path_tuple[-1]
            visited = set(path_tuple)
            if any(nb not in visited for nb in rem_adj[current]):
                can_extend = True
                break
        if not can_extend:
            break
            
    if not beam:
        return []
        
    # Return the best path by score
    best_path = max(beam, key=lambda x: x[1])[0]
    return list(best_path)
>>>>>>> REPLACE

