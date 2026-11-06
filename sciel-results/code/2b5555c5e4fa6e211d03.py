def find_new_vertex_edges(base_colors, base_n, deadline):
    """Find valid edge colors for a new vertex (index base_n) connected to 0..base_n-1."""
    # For each color c, precompute the conflict graph:
    # G_c has vertices 0..base_n-1, edge (i,j) if color(i,j) == c
    # We need S_c (vertices connected to new vertex with color c) to be independent in G_c
    # And S_0, S_1, S_2, S_3 partition {0, ..., base_n-1}
    
    # Build adjacency lists for each color's conflict graph
    conflict_adj = [set() for _ in range(4)]  # conflict_adj[c] = set of (i,j) pairs with color c
    # Actually, let's build: for each c, adj_c[i] = set of j such that color(i,j) == c
    adj = [set() for _ in range(4)]
    for c in range(4):
        adj[c] = [set() for _ in range(base_n)]
    
    idx = 0
    for i in range(base_n):
        for j in range(i + 1, base_n):
            c = base_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    
    # Now do backtracking: assign colors to vertices 0, 1, ..., base_n-1
    # Assignment: assign[i] = color of edge (new_vertex, i)
    # Constraint: if assign[i] == assign[j] == c, then j not in adj[c][i]
    
    assignment = [-1] * base_n
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == base_n:
            return True
        
        # Try colors in order of least constraining value
        for c in range(4):
            # Check if assigning color c to vertex pos is consistent with previous assignments
            valid = True
            for prev in range(pos):
                if assignment[prev] == c and pos in adj[c][prev]:
                    valid = False
                    break
            if valid:
                assignment[pos] = c
                if backtrack(pos + 1):
                    return True
                assignment[pos] = -1
        return False
    
    if backtrack(0):
        return assignment
    
    return None
