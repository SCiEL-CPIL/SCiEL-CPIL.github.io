<<<<<<< SEARCH
def _find_max_biclique(adj, remaining_edges):
    """Find a large biclique in the subgraph defined by remaining_edges."""
    if not remaining_edges:
        return None, None
    
    # Count how many remaining edges each vertex is incident to
    deg = {}
    for u, v in remaining_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    
    if not deg:
        return None, None
    
    # Try multiple anchor vertices and pick the best biclique
    best_left, best_right, best_size = None, None, 0
    
    # Sort by degree descending, try top candidates
    sorted_vertices = sorted(deg.keys(), key=lambda x: -deg[x])
    max_anchors = min(len(sorted_vertices), 20)
    
    for anchor in sorted_vertices[:max_anchors]:
        # Its neighbors in the remaining edge set
        candidates = set()
        for u, v in remaining_edges:
            if u == anchor:
                candidates.add(v)
            elif v == anchor:
                candidates.add(u)
        
        if not candidates:
            continue
        
        # Greedily build biclique: start with anchor on left, try to add vertices
        left = {anchor}
        right = set(candidates)
        
        # Iteratively refine: keep only right vertices connected to all left, and vice versa
        changed = True
        while changed:
            changed = False
            # Filter right: must be connected to all in left
            new_right = set()
            for v in right:
                if all(v in adj[u] for u in left):
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            # Filter left: must be connected to all in right
            new_left = set()
            for u in left:
                if all(u in adj[v] for v in right):
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        if not left or not right:
            continue
        
        # Try to expand left by adding vertices connected to all of right
        for u in range(P['n']):
            if u not in left and u in deg:
                if all(u in adj[v] for v in right):
                    left.add(u)
        
        # Try to expand right by adding vertices connected to all of left
        for v in range(P['n']):
            if v not in right and v in deg:
                if all(v in adj[u] for u in left):
                    right.add(v)
        
        # Calculate size of this biclique (number of edges it covers)
        size = len(left) * len(right)
        if size > best_size:
            best_size = size
            best_left = list(left)
            best_right = list(right)
    
    return best_left, best_right
=======
def _find_max_biclique(adj, remaining_edges):
    """Find a large biclique in the subgraph defined by remaining_edges."""
    if not remaining_edges:
        return None, None
    
    n = P['n']
    
    # Build adjacency sets restricted to remaining edges
    rem_adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Get vertices with remaining edges
    active = [i for i in range(n) if deg[i] > 0]
    if not active:
        return None, None
    
    # Try multiple anchor vertices and pick the best biclique
    best_left, best_right, best_size = None, None, 0
    
    # Sort by degree descending, try top candidates
    sorted_vertices = sorted(active, key=lambda x: -deg[x])
    max_anchors = min(len(sorted_vertices), 30)
    
    for anchor in sorted_vertices[:max_anchors]:
        # Its neighbors in the remaining edge set
        candidates = set(rem_adj[anchor])
        
        if not candidates:
            continue
        
        # Build biclique: anchor on left, try to add other vertices
        left = {anchor}
        right = set(candidates)
        
        # Iteratively refine: keep only right vertices connected to all left, and vice versa
        changed = True
        while changed:
            changed = False
            # Filter right: must be connected to all in left via remaining edges
            new_right = set()
            for v in right:
                if all(v in rem_adj[u] for u in left):
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            # Filter left: must be connected to all in right via remaining edges
            new_left = set()
            for u in left:
                if all(u in rem_adj[v] for v in right):
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        if not left or not right:
            continue
        
        # Try to expand left by adding vertices connected to all of right
        for u in active:
            if u not in left:
                if all(u in rem_adj[v] for v in right):
                    left.add(u)
        
        # Try to expand right by adding vertices connected to all of left
        for v in active:
            if v not in right:
                if all(v in rem_adj[u] for u in left):
                    right.add(v)
        
        # Final refinement
        changed = True
        while changed:
            changed = False
            new_right = set()
            for v in right:
                if all(v in rem_adj[u] for u in left):
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            new_left = set()
            for u in left:
                if all(u in rem_adj[v] for v in right):
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        if not left or not right:
            continue
        
        # Calculate size of this biclique (number of edges it covers)
        size = len(left) * len(right)
        if size > best_size:
            best_size = size
            best_left = list(left)
            best_right = list(right)
    
    return best_left, best_right
>>>>>>> REPLACE

