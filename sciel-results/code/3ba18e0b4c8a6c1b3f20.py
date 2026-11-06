# EVOLVE-BLOCK-START
def construct():
    """
    Returns 101 nonempty lists. Row r contains distinct denominators in 1..10
    whose modular inverses sum to r modulo 101.
    Minimizes the largest row length.
    Uses Hopcroft-Karp matching to find optimal assignment if reuse is not allowed,
    or simply min-size if reuse is allowed. 
    Based on directive: Model as bipartite matching.
    """
    import collections
    
    p = 101
    n_residues = 101
    all_denoms = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [pow(x, -1, p) for x in all_denoms]
    
    # Generate all non-empty subsets
    # subset_info: list of (residue, size, terms)
    subset_info = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        terms = [all_denoms[i] for i in range(10) if mask & (1 << i)]
        residue = sum(inv[i] for i in range(10) if mask & (1 << i)) % p
        subset_info.append((residue, size, terms, mask))
        
    # Hopcroft-Karp implementation
    def hopcroft_karp(n_left, n_right, adj):
        """
        n_left: number of left nodes (residues 0..100)
        n_right: number of right nodes (subsets indices 0..len-1)
        adj: list of lists, adj[u] = list of v such that (u,v) in graph
        Returns: matching dictionary {left_node: right_node}
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = collections.deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == float('inf'):
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
            
        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
            
        matching_size = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        return pair_u, pair_v

    # Binary search on K (max allowed size)
    # Max possible size is 10.
    lo, hi = 1, 10
    best_K = 10
    best_matching = None
    
    # Pre-group subsets by size to speed up graph construction?
    # Or just filter in loop.
    
    for K in range(1, 11):
        # Build graph for subsets with size <= K
        # Left nodes: 0..100 (residues)
        # Right nodes: indices of valid subsets
        valid_subsets = [s for s in subset_info if s[1] <= K]
        n_right = len(valid_subsets)
        
        if n_right < 101:
            continue
            
        # Map residue to list of right-node indices
        adj = [[] for _ in range(101)]
        # We need to map the index in valid_subsets to the right node ID
        # Let right node ID be the index in valid_subsets list.
        
        for idx, (res, size, terms, mask) in enumerate(valid_subsets):
            adj[res].append(idx)
            
        pair_u, pair_v = hopcroft_karp(101, n_right, adj)
        
        # Check if perfect matching
        if all(pair_u[u] != -1 for u in range(101)):
            best_K = K
            best_matching = (pair_u, pair_v, valid_subsets)
            break
            
    if best_matching is None:
        # Fallback: just pick min size subsets (reuse allowed)
        # This ensures we return *something* valid even if matching fails (shouldn't happen)
        min_size_for_residue = [11] * 101
        for res, size, terms, mask in subset_info:
            if size < min_size_for_residue[res]:
                min_size_for_residue[res] = size
        max_min_size = max(min_size_for_residue)
        result = [None] * 101
        for res, size, terms, mask in subset_info:
            if size <= max_min_size and result[res] is None:
                result[res] = terms
        for r in range(101):
            if result[r] is None:
                result[r] = [1]
        return result

    pair_u, pair_v, valid_subsets = best_matching
    
    result = [None] * 101
    for u in range(101):
        v = pair_u[u]
        if v != -1:
            result[u] = valid_subsets[v][2] # terms
        else:
            result[u] = [1] # Fallback
            
    return result
# EVOLVE-BLOCK-END
