<<<<<<< SEARCH
def _build():
    N = 50
    adj = _neighbors_grid(N)
    # Use sets for faster membership check
    adj_set = [set(nei) for nei in adj]
    comp_adj = []
    for i in range(N):
        comp_nei = [j for j in range(N) if j != i and j not in adj_set[i]]
        comp_adj.append(comp_nei)
        
    deadline = time.time() + 100.0 # Leave time for verification
    
    # Try multiple random seeds for robustness
    for seed in range(50):
        if time.time() > deadline:
            break
        cg = _proper_4color(adj, N, deadline, seed)
        if cg is None:
            continue
        cc = _proper_4color(comp_adj, N, deadline, seed + 100)
        if cc is None:
            continue
            
        # Construct edge colours
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                if j in adj_set[i]:
                    colors.append(cg[i])
                else:
                    colors.append(cc[i])
        
        # Verify
        if _verify(N, colors):
            return N, colors
            
    return None
=======
def _build():
    N = 50
    adj = _neighbors_grid(N)
    # Use sets for faster membership check
    adj_set = [set(nei) for nei in adj]
    comp_adj = []
    for i in range(N):
        comp_nei = [j for j in range(N) if j != i and j not in adj_set[i]]
        comp_adj.append(comp_nei)
        
    deadline = time.time() + 100.0 # Leave time for verification
    
    # Try multiple random seeds for robustness
    for seed in range(100):
        if time.time() > deadline:
            break
        cg = _proper_4color(adj, N, deadline, seed)
        if cg is None:
            continue
        cc = _proper_4color(comp_adj, N, deadline, seed + 100)
        if cc is None:
            continue
            
        # Construct edge colours
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                if j in adj_set[i]:
                    colors.append(cg[i])
                else:
                    colors.append(cc[i])
        
        # Verify
        if _verify(N, colors):
            return N, colors
            
    # Fallback: try direct 4-coloring of complete graph using local search
    # This is less likely to succeed but worth trying
    full_adj = [[j for j in range(N) if j != i] for i in range(N)]
    for seed in range(20):
        if time.time() > deadline:
            break
        result = _local_search_4color(full_adj, N, deadline, seed + 1000)
        if result is not None:
            # Convert vertex coloring to edge coloring
            colors = []
            for i in range(N):
                for j in range(i + 1, N):
                    # Use a deterministic mapping from vertex colors to edge colors
                    # This won't work directly; we need a proper construction
                    pass
            # Skip this fallback as it's not straightforward
            break
    
    return None
>>>>>>> REPLACE
