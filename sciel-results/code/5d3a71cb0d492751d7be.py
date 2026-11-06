import time

def construct():
    start_time = time.time()
    
    # Try to build colorings from largest n down
    for n in range(27, 10, -1):
        if time.time() - start_time > 115:
            break
        
        # Greedy constructive with backtracking
        # colors[i] stores the color assigned to vertex i (for i >= 1, relative to vertex 0)
        # Actually, let's store edge colors directly
        # adj[c][v] = set of neighbors of v in color c
        
        # We build vertex by vertex. For each new vertex v (starting from 1),
        # we need to assign a color to each edge (u, v) for u < v.
        # Constraint: for each color c, no two vertices can have two common neighbors in color c.
        # When adding vertex v, for each color c, the neighbors of v in color c must form
        # a set such that no two vertices in that set already share a common neighbor in color c
        # (other than v itself, which doesn't count since v is new).
        
        # More precisely: if u1 and u2 are both neighbors of v in color c,
        # then N_c(u1) ∩ N_c(u2) must be empty (excluding v, but v isn't in any N_c yet).
        # Also, for any u1 that is a neighbor of v in color c, and any u2 that is NOT a neighbor
        # of v in color c, we need N_c(u1) ∩ N_c(u2) ∩ {v} = empty, which is automatically
        # satisfied since v is new.
        
        # So the constraint when assigning colors to edges (u, v) for u < v is:
        # For each color c, let S_c = {u : color(u,v) = c}. Then for any u1, u2 in S_c,
        # N_c(u1) ∩ N_c(u2) must be empty.
        
        # This is equivalent to: the sets N_c(u) for u in S_c must be pairwise disjoint.
        
        def try_build(n, max_time=110):
            # adj[c][v] = set of neighbors of v in color c
            adj = [[set() for _ in range(n)] for _ in range(5)]
            # colors[i][j] for i < j, stored as list of lists
            colors = [[0] * n for _ in range(n)]
            
            # Vertex 0 has no edges to assign
            # For vertex v = 1, 2, ..., n-1:
            #   assign color to each edge (u, v) for u < v
            
            def backtrack(v):
                if time.time() - start_time > max_time:
                    return False
                
                if v == n:
                    return True
                
                # Assign colors to edges (0,v), (1,v), ..., (v-1,v)
                # For each color c, track which vertices u have been assigned color c for edge (u,v)
                # Constraint: for each color c, the set of u's assigned color c must have
                # pairwise disjoint N_c(u) sets.
                
                # We'll use backtracking on the edges in order
                def assign_edge(u_idx):
                    if u_idx == v:
                        return backtrack(v + 1)
                    
                    for c in range(5):
                        # Check if assigning color c to edge (u_idx, v) is valid
                        # We need to check that for all u' < u_idx that already have color c
                        # assigned to edge (u', v), N_c(u') ∩ N_c(u_idx) is empty.
                        # Also need to check against all existing neighbors.
                        
                        valid = True
                        for u2 in range(u_idx):
                            if colors[u2][v] == c:
                                # u2 and u_idx both would be neighbors of v in color c
                                # Check N_c(u2) ∩ N_c(u_idx)
                                if adj[c][u2] & adj[c][u_idx]:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        # Assign
                        colors[u_idx][v] = c
                        colors[v][u_idx] = c
                        adj[c][u_idx].add(v)
                        adj[c][v].add(u_idx)
                        
                        if assign_edge(u_idx + 1):
                            return True
                        
                        # Unassign
                        adj[c][u_idx].discard(v)
                        adj[c][v].discard(u_idx)
                        colors[u_idx][v] = 0
                        colors[v][u_idx] = 0
                    
                    return False
                
                return assign_edge(0)
            
            return backtrack(1), adj, colors
        
        # Try with some heuristic ordering
        found, adj, colors = try_build(n)
        if found:
            # Extract colors in lexicographic order
            color_list = []
            for i in range(n):
                for j in range(i+1, n):
                    color_list.append(colors[i][j])
            return {'n': n, 'colors': color_list}
    
    # Fallback: return a small valid coloring
    # n=2 is trivially valid
    return {'n': 2, 'colors': [0]}
