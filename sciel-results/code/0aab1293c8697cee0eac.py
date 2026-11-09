import time
import random

def has_path_len_4(adj, u, v):
    """
    Check if there is a path of length exactly 4 from u to v in the graph defined by adj.
    adj is a list of sets (or bitmasks) for each vertex.
    """
    # BFS or DFS to depth 4
    # Since graph is small (<=17), this is fast.
    # We need a path u -> a -> b -> c -> v.
    # We can iterate over all pairs (a,b) such that (u,a) and (b,v) are edges? No.
    # Path: u-a-b-c-v.
    # Edges: (u,a), (a,b), (b,c), (c,v).
    # We can check if there exists a,b,c such that:
    # a in adj[u] and b in adj[a] and c in adj[b] and v in adj[c].
    
    # Optimization: 
    # Let N1 = adj[u]
    # Let N2 = {x | exists y in N1 such that x in adj[y]}
    # Let N3 = {x | exists y in N2 such that x in adj[y]}
    # We need v in N4? No, N4 is neighbors of N3.
    # Actually, we just need to see if v is reachable in exactly 4 steps.
    
    # Simple DFS with depth limit 4.
    # State: (current_node, depth)
    # Start: (u, 0)
    # Goal: reach v at depth 4.
    
    # To avoid cycles, we can't just do simple BFS if we care about "exact length".
    # But in a simple graph, a path of length 4 is a sequence of 5 distinct vertices?
    # The task says "exact search for not necessarily induced cycles".
    # A cycle is a closed path with distinct vertices (except start/end).
    # So the path $a \dots b$ of length 4 in the $C_6$ check must be a simple path (distinct vertices).
    # So we need a simple path of length 4 from u to v.
    
    # Given max n=17, a simple DFS checking for simple paths of length 4 is very fast.
    
    def dfs(curr, target, depth, visited):
        if depth == 4:
            return curr == target
        for next_node in adj[curr]:
            if next_node not in visited:
                visited.add(next_node)
                if dfs(next_node, target, depth + 1, visited):
                    return True
                visited.remove(next_node)
        return False
    
    return dfs(u, v, 0, {u})

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Seed solution for n=9
    # n=9, colors based on min(j-i, n-(j-i)) - 1
    # Distances: 1,2,3,4. Colors 0,1,2,3.
    # This creates 4 cycles?
    # Let's verify the seed is valid.
    # The seed is provided and scored 1.0, so it is valid.
    
    # We will try to extend to n=17.
    # We maintain a list of colors for edges.
    # colors[i][j] for i<j.
    
    # To allow backtracking, we might need to store the full state.
    # However, extending vertex by vertex is the most promising.
    
    # Let's try a randomized restart strategy.
    # In each restart, we try to build a coloring for n=17.
    # We fix the first 9 vertices to the seed coloring.
    # Then we try to assign colors to edges involving vertices 9..16.
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Precompute adjacencies for the seed to speed up checks?
    # The adjacencies change as we add vertices.
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        # Start with seed
        n_current = 9
        # colors is a flat list.
        # Index for (i,j) is sum_{k=0}^{i-1} (n_current - k - 1) + (j - i - 1) ?
        # Easier: use a 2D list or dict.
        # For n=17, 136 edges. A list of length 136 is fine.
        # But extending changes the indices.
        # Let's use a 2D array `edge_color[i][j]` for i<j.
        
        edge_color = [[-1]*17 for _ in range(17)]
        
        # Initialize with seed
        for i in range(9):
            for j in range(i+1, 9):
                edge_color[i][j] = min(j - i, 9 - j + i) - 1
                edge_color[j][i] = edge_color[i][j]
                
        # Try to extend to 17
        success = True
        # We need to assign colors for edges (v, u) where v >= 9 and u < v.
        # Order: v from 9 to 16.
        
        # We use backtracking for the extension part.
        # Since 9 is fixed, we only backtrack on vertices 9..16.
        
        def backtrack(v):
            nonlocal best_n, best_colors
            if v == 17:
                # Success!
                # Flatten colors
                cols = []
                for i in range(17):
                    for j in range(i+1, 17):
                        cols.append(edge_color[i][j])
                best_n = 17
                best_colors = cols
                return True
            if time.time() > deadline:
                return False
                
            # Assign colors to (v, u) for u in 0..v-1
            # To make it faster, we can randomize the order of u?
            # Or just iterate.
            
            # We need to check constraints as we assign.
            # When we assign edge (v, u) = c, we must check for C6.
            # A C6 involving (v, u) must have another edge (v, w) of color c.
            # So if we assign (v, u) = c, we check all previously assigned (v, w) with color c.
            # If such w exists, check if path_len_4(u, w, G_c) exists.
            
            # We also need to check C6s that don't involve v? No, those were already checked.
            # We only need to check new C6s formed by adding edges incident to v.
            
            # Optimization: Precompute adj lists for each color for the current partial graph?
            # The graph grows.
            
            # Let's just do the check on the fly.
            
            # To improve search, try colors in a randomized order?
            # Or use a heuristic: try colors that don't conflict.
            
            # For simplicity, try 0,1,2,3.
            # If we fail, we backtrack.
            
            # We need to store the assignments for v to backtrack.
            # Assignments for v are edge_color[v][0..v-1].
            
            # We can use a recursive function that assigns u from v-1 down to 0?
            # Or u from 0 to v-1.
            
            # Let's assign u from v-1 down to 0.
            # This way, when we assign (v, u), all (v, w) for w>u are already assigned.
            # We only need to check against w>u.
            
            def assign_u(u):
                if u < 0:
                    return backtrack(v+1)
                
                # Try colors 0..3
                for c in range(4):
                    # Check constraint: 
                    # For all w in u+1..v-1 such that edge_color[v][w] == c:
                    #   if has_path_len_4(u, w, G_c): return False
                    valid = True
                    # Build G_c adj list? 
                    # G_c consists of edges (i,j) where i,j < v and edge_color[i][j]==c, 
                    # PLUS edges (v, w) where w < v and edge_color[v][w]==c.
                    # Wait, the path is in the graph of color c.
                    # The path is between u and w.
                    # The path vertices must be distinct.
                    # The path cannot use v? 
                    # The C6 is v-u-...-w-v.
                    # The path u...w is in the subgraph of vertices 0..v-1?
                    # No, the path is in the whole graph, but since it's a C6, 
                    # and v is only connected to u and w in this cycle, 
                    # the path u...w cannot pass through v.
                    # So the path is in the subgraph induced by {0, ..., v-1}.
                    
                    # So we only need the adj list of the subgraph {0, ..., v-1} for color c.
                    
                    # We can maintain adj lists for each color for the current prefix.
                    # But we are backtracking, so maintaining them is tricky.
                    # Given n<=17, rebuilding adj list is cheap.
                    
                    # Let's build adj list for color c for vertices 0..v-1.
                    adj_c = [[] for _ in range(v)]
                    for i in range(v):
                        for j in range(i+1, v):
                            if edge_color[i][j] == c:
                                adj_c[i].append(j)
                                adj_c[j].append(i)
                    
                    # Now check for all w in u+1..v-1 with edge_color[v][w]==c
                    for w in range(u+1, v):
                        if edge_color[v][w] == c:
                            if has_path_len_4(adj_c, u, w):
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Assign
                    edge_color[v][u] = c
                    edge_color[u][v] = c
                    
                    if assign_u(u-1):
                        return True
                    else:
                        edge_color[v][u] = -1
                        edge_color[u][v] = -1
                return False
            
            if assign_u(v-1):
                return True
            return False
        
        if backtrack(9):
            break # Found solution for 17
        
    # If we didn't find 17, return best found (likely 9)
    # Note: The above logic only tries to extend the seed.
    # If seed extension fails, we might miss other n=10..16 colorings.
    # But 17 is the target.
    
    return {'n': best_n, 'colors': best_colors}
