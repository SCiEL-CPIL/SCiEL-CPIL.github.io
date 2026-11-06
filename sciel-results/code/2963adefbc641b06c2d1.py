"""Program using Hopcroft-Karp bipartite matching to minimize max row length."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns 101 nonempty lists. Row r contains distinct denominators in 1..10
    whose modular inverses sum to r modulo 101.
    Minimizes the largest row length.
    Uses bipartite matching to ensure distinct subsets are assigned if possible,
    falling back to independent selection if matching fails (though matching should succeed for small K).
    """
    p = 101
    n_residues = 101
    all_denoms = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [pow(x, -1, p) for x in all_denoms]
    
    # Generate all non-empty subsets of {1..10}
    # subset_data: list of (residue, size, terms_list)
    subset_data = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        terms = [all_denoms[i] for i in range(10) if mask & (1 << i)]
        residue = sum(inv[i] for i in range(10) if mask & (1 << i)) % p
        subset_data.append((residue, size, terms))
    
    # Hopcroft-Karp implementation
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[u] contains v's connected to u.
        n_left: number of left nodes (0..n_left-1)
        n_right: number of right nodes (0..n_right-1)
        Returns matching: list of length n_left, matching[u] = v or -1
        """
        import collections
        
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
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        return pair_u, matching

    # Binary search on K (max row length)
    # K must be at least 1. Max possible is 10.
    low, high = 1, 10
    best_k = 10
    best_matching = None
    
    # To speed up, we can just iterate K from 1 to 10 and pick the first that works.
    for k in range(1, 11):
        # Filter subsets with size <= k
        # Left nodes: residues 0..100
        # Right nodes: indices of subsets with size <= k
        
        # Build adjacency: residue -> list of subset_indices
        # We need to map subset indices to right-node indices.
        # Let's just use the original index in subset_data for right nodes, 
        # but we need to handle n_right correctly for HK.
        # It's easier to re-index the right nodes for the current K.
        
        valid_subsets = []
        subset_to_right_idx = {}
        for i, (res, sz, terms) in enumerate(subset_data):
            if sz <= k:
                subset_to_right_idx[i] = len(valid_subsets)
                valid_subsets.append((res, i, terms))
        
        n_right = len(valid_subsets)
        
        # Build adj for HK
        # adj[res] = list of right_indices
        adj = [[] for _ in range(n_residues)]
        for res, orig_idx, terms in valid_subsets:
            right_idx = subset_to_right_idx[orig_idx]
            adj[res].append(right_idx)
        
        # Run HK
        pair_u, match_count = hopcroft_karp(adj, n_residues, n_right)
        
        if match_count == n_residues:
            best_k = k
            best_matching = pair_u
            # We need to map back to terms
            # pair_u[r] is the right_idx
            # We need to find the terms corresponding to right_idx
            # Let's store the terms in a list indexed by right_idx
            terms_by_right_idx = [None] * n_right
            for res, orig_idx, terms in valid_subsets:
                terms_by_right_idx[subset_to_right_idx[orig_idx]] = terms
            
            result = [None] * n_residues
            for r in range(n_residues):
                if pair_u[r] != -1:
                    result[r] = terms_by_right_idx[pair_u[r]]
                else:
                    # Should not happen if match_count == n_residues
                    result[r] = [1]
            return result
    
    # Fallback if no K worked (should not happen)
    # Just return independent best subsets
    min_size_for_residue = [11] * 101
    for res, sz, terms in subset_data:
        if sz < min_size_for_residue[res]:
            min_size_for_residue[res] = sz
            # Store terms too? No, just find them later.
    
    max_min_size = max(min_size_for_residue)
    result = [None] * 101
    for res, sz, terms in subset_data:
        if sz <= max_min_size and result[res] is None:
            result[res] = terms
    
    for r in range(101):
        if result[r] is None:
            result[r] = [1]
            
    return result
# EVOLVE-BLOCK-END
