def construct():
    import time
    import random
    from itertools import combinations
    
    # Locate the input data in the global namespace
    P = None
    for name in ['P', 'p', 'data', 'graph', 'G', 'input', 'params', 'problem']:
        if name in globals():
            P = globals()[name]
            break

    if P is None:
        # Fallback if input data is not found in expected globals
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin

    n = P['n']
    edges = P['edges']

    # Build adjacency list and set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    # Generate candidate paths
    # We look for simple paths of length 1, 2, and 3
    # Length 1: Edges
    # Length 2: u-v-w where u!=w and v is neighbor of both
    # Length 3: u-v-w-x where u!=x and consecutive edges exist
    
    candidates = []
    seen_paths = set()
    
    # Helper to add a path if unique
    def add_path(path):
        # Normalize path: sort endpoints, but keep path order?
        # The path itself matters for interiors.
        # But (u, ..., v) is same pair as (v, ..., u).
        # We can store the tuple of the path.
        # To avoid duplicates like [1,2,3] and [3,2,1], we can sort the endpoints for a key,
        # but we need to keep the actual path vertices for interior calculation.
        # Actually, [1,2,3] and [3,2,1] have the same interior {2} and same endpoints {1,3}.
        # So we can just pick one orientation.
        key = tuple(sorted(path))
        if key in seen_paths:
            return
        seen_paths.add(key)
        interior = set(path[1:-1])
        endpoints = {path[0], path[-1]}
        candidates.append({
            'path': path,
            'interior': interior,
            'endpoints': endpoints
        })

    # Length 1
    for u in range(n):
        for v in adj[u]:
            if u < v: # Avoid duplicates
                add_path([u, v])

    # Length 2
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w == u:
                    continue
                if u < w: # Avoid duplicates (u,w) and (w,u)
                    # Check if direct edge exists? No, path is simple.
                    # If u-w is an edge, [u,w] is already added.
                    # [u,v,w] is a distinct path object.
                    add_path([u, v, w])

    # Length 3
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    if u < x:
                        add_path([u, v, w, x])

    # If no candidates (should not happen), fallback
    if not candidates:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}

    best_branches = set()
    best_paths = []
    best_size = 0
    
    # Greedy search with multiple restarts
    num_restarts = 200
    random.seed(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_branches = set()
        current_interiors = set()
        selected_paths = []
        
        # List of indices of candidates still "available"
        # We don't remove them, we just check validity
        # But for efficiency, we can filter
        
        while True:
            if time.time() > deadline:
                break
                
            best_gain = 0
            best_cand_indices = []
            
            for i, cand in enumerate(candidates):
                # Check validity
                # Interior must be disjoint from current_interiors
                if cand['interior'] & current_interiors:
                    continue
                # Interior must be disjoint from current_branches
                if cand['interior'] & current_branches:
                    continue
                
                # Calculate gain
                new_branches = cand['endpoints'] - current_branches
                gain = len(new_branches)
                
                if gain > best_gain:
                    best_gain = gain
                    best_cand_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_cand_indices.append(i)
            
            if best_gain == 0:
                break
            
            # Pick randomly from best candidates
            idx = random.choice(best_cand_indices)
            cand = candidates[idx]
            
            selected_paths.append(cand['path'])
            current_branches |= cand['endpoints']
            current_interiors |= cand['interior']
            
        # Update best
        if len(current_branches) > best_size:
            best_size = len(current_branches)
            best_branches = set(current_branches)
            best_paths = [list(p) for p in selected_paths]
            
            # Early exit if we found all vertices?
            if best_size == n:
                break

    # Fallback
    if not best_paths:
        if edges:
            u, v = edges[0]
            best_branches = {int(u), int(v)}
            best_paths = [[int(u), int(v)]]

    # Format output
    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    
    return {'branches': branches_out, 'paths': paths_out}
