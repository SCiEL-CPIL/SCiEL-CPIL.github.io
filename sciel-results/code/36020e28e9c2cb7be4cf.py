import time

def construct():
    start_time = time.time()
    deadline = 100.0  # seconds, leaving margin for 120s limit
    
    # 1. Construct Polarity Graph for q=5
    q = 5
    # Generate points of projective plane PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for y in range(q):
        points.append((0, 1, y))
    points.append((0, 0, 1))
    
    # Canonicalize: first non-zero is 1. 
    # The generation above already does this mostly, but let's ensure uniqueness.
    # (1, x, y) covers 25 points.
    # (0, 1, y) covers 5 points.
    # (0, 0, 1) covers 1 point.
    # Total 31.
    
    n_pol = len(points)
    # Adjacency matrix or list
    # u ~ v if u[0]*v[1] + u[1]*v[0] + u[2]*v[2] == 0
    adj = [[False]*n_pol for _ in range(n_pol)]
    for i in range(n_pol):
        for j in range(i+1, n_pol):
            u = points[i]
            v = points[j]
            val = (u[0]*v[1] + u[1]*v[0] + u[2]*v[2]) % q
            if val == 0:
                adj[i][j] = True
                adj[j][i] = True
                
    # Check C4 freeness? 
    # Polarity graphs are C4-free by theory. 
    # But let's assume it is.
    
    # 2. Determine max k triangles to add
    # We want to find max k in [0, 4] such that alpha_pol + k < 11
    # i.e., alpha_pol < 11 - k
    # Let's check for k=4, 3, 2, 1, 0
    
    def has_independent_set_size(M, adj, n):
        # Backtracking search for independent set of size M
        # n is small (31)
        if M <= 0: return True
        if n < M: return False
        
        # Heuristic: pick vertex with lowest degree in current subgraph?
        # Or just simple backtracking
        # To speed up, we can use a greedy coloring bound?
        # Let's try a simple recursive search with pruning.
        
        # Order vertices by degree
        degrees = [sum(adj[i][j] for j in range(n) if j != i) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x])
        
        # Map old index to new index
        # Actually, let's just use the original indices but prune.
        
        def backtrack(start_idx, current_set, remaining_needed):
            if remaining_needed == 0:
                return True
            if len(current_set) + (n - start_idx) < remaining_needed:
                return False
            
            # Pruning: if remaining_needed > number of vertices left that are not adjacent to current_set?
            # Simple: iterate from start_idx
            for i in range(start_idx, n):
                # Check if i is adjacent to any in current_set
                is_independent = True
                for v in current_set:
                    if adj[v][i]:
                        is_independent = False
                        break
                if not is_independent:
                    continue
                
                # Try adding i
                if backtrack(i+1, current_set + [i], remaining_needed - 1):
                    return True
            return False
        
        return backtrack(0, [], M)

    best_k = 0
    for k in range(4, -1, -1):
        target_alpha = 11 - k
        # We need alpha_pol < target_alpha
        # So we check if there is an independent set of size target_alpha
        # If there IS, then alpha_pol >= target_alpha, so this k is too large.
        # If there IS NOT, then alpha_pol < target_alpha, so this k is valid.
        
        if time.time() > deadline:
            break
            
        # Check if independent set of size target_alpha exists
        exists = has_independent_set_size(target_alpha, adj, n_pol)
        if not exists:
            best_k = k
            break
            
    # Construct final graph
    # Vertices 0..n_pol-1 are the polarity graph
    # Vertices n_pol .. n_pol + 3*best_k - 1 are the triangles
    n_total = n_pol + 3 * best_k
    edges = []
    
    # Add polarity graph edges
    for i in range(n_pol):
        for j in range(i+1, n_pol):
            if adj[i][j]:
                edges.append([i, j])
                
    # Add triangles
    for t in range(best_k):
        a = n_pol + 3*t
        b = n_pol + 3*t + 1
        c = n_pol + 3*t + 2
        edges.append([a, b])
        edges.append([a, c])
        edges.append([b, c])
        
    return {'n': n_total, 'edges': edges}
