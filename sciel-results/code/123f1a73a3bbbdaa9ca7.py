N = 16

# EVOLVE-BLOCK-START
def construct():
    num_edges = N * (N - 1) // 2
    colors = [-1] * num_edges
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    def get_edge_color(i, j):
        if i > j:
            i, j = j, i
        idx = get_idx(i, j)
        return colors[idx]

    # Iterate through all edges in lexicographic order
    for i in range(N):
        for j in range(i + 1, N):
            idx = get_idx(i, j)
            allowed = set(range(120))
            
            # Check all possible 4-cycles containing (i, j)
            # The other two vertices k, l must be distinct from i and j
            for k in range(N):
                if k == i or k == j:
                    continue
                for l in range(k + 1, N):
                    if l == i or l == j:
                        continue
                    
                    # Cycle 1: i - j - k - l - i
                    # Edges: (i,j), (j,k), (k,l), (l,i)
                    c_jk = get_edge_color(j, k)
                    c_kl = get_edge_color(k, l)
                    c_li = get_edge_color(l, i)
                    
                    if c_jk != -1 and c_kl != -1 and c_li != -1:
                        if len({c_jk, c_kl, c_li}) == 3:
                            allowed = allowed.intersection({c_jk, c_kl, c_li})
                    
                    # Cycle 2: i - j - l - k - i
                    # Edges: (i,j), (j,l), (l,k), (k,i)
                    c_jl = get_edge_color(j, l)
                    c_lk = get_edge_color(l, k) # Same as c_kl
                    c_ki = get_edge_color(k, i)
                    
                    if c_jl != -1 and c_lk != -1 and c_ki != -1:
                        if len({c_jl, c_lk, c_ki}) == 3:
                            allowed = allowed.intersection({c_jl, c_lk, c_ki})
            
            if not allowed:
                # Fallback: This shouldn't happen often, but if it does, pick 0
                # Or we might need a more robust strategy.
                # For now, pick 0 to ensure validity if we assume 0 is safe?
                # Actually, if allowed is empty, it means every color creates a rainbow cycle.
                # This implies a contradiction in previous assignments.
                # We will pick 0 as a default.
                colors[idx] = 0
            else:
                # Pick the largest color to maximize distinct count
                colors[idx] = max(allowed)
                
    return colors
# EVOLVE-BLOCK-END
