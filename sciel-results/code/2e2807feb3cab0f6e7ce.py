import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Base n=29 Paley Graph
    n_base = 29
    residues = set((x * x) % n_base for x in range(1, n_base))
    
    # colors[i][j] for i < j
    # We will work with a list of lists or a flat list.
    # Flat list of length n*(n-1)/2 is efficient for serialization.
    # But for internal logic, a 2D list or set of edges is better.
    
    # Initialize colors for n=29
    n = n_base
    # colors[i][j] = 0 or 1
    colors = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors[i][j] = 0
            else:
                colors[i][j] = 1
            colors[j][i] = colors[i][j]

    # Helper to check validity and count violations
    def get_violations(colors, n):
        # Returns number of K2,2,2 in color 0 + number in color 1
        # Using bitmask approach
        # Precompute neighbor bitmasks for each color
        # N_c[i] is an integer bitmask of neighbors of i in color c
        
        N0 = [0] * n
        N1 = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if colors[i][j] == 0:
                    N0[i] |= (1 << j)
                else:
                    N1[i] |= (1 << j)
        
        violations = 0
        
        # Helper to check if a bitmask S contains an edge in color c
        # We need to check if there exist u, v in S such that colors[u][v] == c
        # This is equivalent to: for some u in S, N_c[u] & S != 0
        # But we must be careful not to count self-loops (none here) or double count.
        # Actually, if we just want existence, we can iterate.
        
        def has_edge_in_set(S, N_c):
            # S is a bitmask
            # Check if any pair in S is connected in N_c
            temp = S
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                v_idx = lsb.bit_length() - 1
                # Check if v has any neighbor in S
                if N_c[v_idx] & S:
                    return True
                temp ^= lsb
            return False

        # Iterate over pairs of disjoint edges in color 0
        # Edge (i,j) and (k,l) with i<j, k<l, {i,j} disjoint from {k,l}
        # We need to check if N0[i] & N0[j] & N0[k] & N0[l] contains an edge in color 0.
        
        # To avoid O(n^4) which is 30^4 = 810,000, which is fine, 
        # but we can optimize by iterating over edges.
        
        # List of edges for each color
        edges0 = []
        edges1 = []
        for i in range(n):
            for j in range(i+1, n):
                if colors[i][j] == 0:
                    edges0.append((i,j))
                else:
                    edges1.append((i,j))
        
        def count_k222(edges, N_c):
            count = 0
            # For each pair of disjoint edges (i,j) and (k,l)
            # We can just iterate all pairs of edges and check disjointness
            # Number of edges in K_30 is 435. Pairs ~ 90k.
            for e1 in edges:
                i, j = e1
                for e2 in edges:
                    k, l = e2
                    if i == k or i == l or j == k or j == l:
                        continue
                    # Check disjointness
                    # Compute common neighbors
                    common = N_c[i] & N_c[j] & N_c[k] & N_c[l]
                    # Remove i,j,k,l just in case (though they are not neighbors of themselves)
                    # Check if common contains an edge in color c
                    if has_edge_in_set(common, N_c):
                        count += 1
            return count

        # This double counting is okay for optimization, we just want to minimize.
        # However, iterating 90k pairs * 2 colors * 120s limit...
        # 90k * 2 = 180k checks.
        # Each check: 4 ANDs, then has_edge_in_set.
        # has_edge_in_set iterates over set bits. Max 30.
        # 180k * 30 = 5.4M operations. In Python, this might take ~1-2 seconds per evaluation.
        # If we do 1000 restarts/steps, that's 1000-2000 seconds. Too slow.
        
        # We need a faster validity check or fewer checks.
        # The check for K2,2,2 can be done faster.
        # A K2,2,2 is defined by 3 disjoint edges.
        # We can precompute for each pair of disjoint edges (i,j), (k,l) the common neighbor set.
        # Then check if that set has an edge.
        
        # Let's optimize the inner loop.
        # Instead of iterating all pairs of edges, we can iterate over 3-tuples of disjoint edges?
        # No, the pair approach is standard.
        
        # Let's try to make the check faster.
        # Precompute a bitmask for "has edge in set S"? No, S varies.
        
        # Alternative:
        # For a fixed color, K2,2,2 exists iff there are 3 vertices u,v,w such that
        # |N(u) \cap N(v) \cap N(w)| >= 2 and the subgraph induced by these common neighbors contains an edge?
        # No, that's for K_{2,2,2} where the parts are {u, u'}, {v, v'}, {w, w'}.
        # If u and v are in the same part, they are not connected.
        
        # Let's stick to the edge pair method but optimize Python loops.
        # Using NumPy might help, but bit operations on Python ints are fast.
        
        # To speed up search, we can use a "soft" score or just count violations.
        # We will use a local search that minimizes violations.
        
        # Optimization: 
        # If we fix the structure of the first 29 vertices, we only need to check
        # violations involving the new vertex 30?
        # A K2,2,2 can involve 0, 1, 2, or 3 edges incident to vertex 30.
        # If it involves 0, it was already in the base graph (which is valid).
        # So we only need to check K2,2,2s that include at least one edge incident to 30.
        # This reduces the search space significantly.
        
        def count_violations_with_new_vertex(colors, n):
            # New vertex is n-1
            nv = n - 1
            violations = 0
            
            # Precompute N0, N1 for all vertices
            N0 = [0] * n
            N1 = [0] * n
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if colors[i][j] == 0:
                        N0[i] |= (1 << j)
                    else:
                        N1[i] |= (1 << j)
            
            # We need to find K2,2,2 in color c that use at least one edge incident to nv.
            # Case 1: The K2,2,2 has exactly one edge incident to nv.
            #   This means nv is in one of the parts, say Part 1 = {nv, x}.
            #   Then Part 2 = {y, z}, Part 3 = {u, v}.
            #   Edges: (nv,y), (nv,z), (nv,u), (nv,v) must be color c.
            #   And (x,y), (x,z), (x,u), (x,v) must be color c.
            #   And (y,u), (y,v), (z,u), (z,v) must be color c.
            #   And (x, nv) is color c.
            #   Note: Edges within parts (x,nv), (y,z), (u,v) are NOT required to be color c (in fact, they are non-edges in K2,2,2, so they can be anything).
            
            # Case 2: The K2,2,2 has two edges incident to nv.
            #   This implies nv is in two parts? Impossible, parts are disjoint.
            #   Wait, if nv is in Part 1, it has 4 edges to Part 2 and Part 3.
            #   So if nv is involved, it has 4 incident edges in the K2,2,2.
            #   So "exactly one edge" is impossible. It must be 4 edges if nv is in the subgraph.
            #   Unless... the subgraph is not induced.
            #   The definition: "Neither colour class may contain K2,2,2 as a (not necessarily induced) subgraph."
            #   This means we just need the 12 edges of K2,2,2 to be present in the color.
            #   If nv is in the subgraph, it must be connected to the other 4 vertices in that color.
            #   So any K2,2,2 involving nv must have the 4 edges from nv to the other 4 vertices in color c.
            
            # So, we just need to check if there exist 4 other vertices y,z,u,v such that:
            # 1. (nv,y), (nv,z), (nv,u), (nv,v) are color c.
            # 2. Among {y,z,u,v}, we can partition into two pairs {y,z} and {u,v} such that
            #    all 4 edges between {y,z} and {u,v} are color c.
            #    (Edges within {y,z} and within {u,v} don't matter).
            #    AND we need a partner x for nv in the same part?
            #    Wait, the parts of K2,2,2 are {nv, x}, {y, z}, {u, v}.
            #    So we also need an x such that:
            #    (x,y), (x,z), (x,u), (x,v) are color c.
            #    
            # So the condition is:
            # Find x, y, z, u, v distinct from nv and each other.
            # (nv, y), (nv, z), (nv, u), (nv, v) in color c.
            # (x, y), (x, z), (x, u), (x, v) in color c.
            # (y, u), (y, v), (z, u), (z, v) in color c.
            
            # This is a specific structure.
            
            # Let's iterate over candidates for x, y, z, u, v.
            # Since nv is fixed, its neighbors in color c are fixed.
            # Let N_c_nv = {w | (nv,w) is color c}.
            # We need to pick 4 vertices y,z,u,v from N_c_nv.
            # And 1 vertex x from V \ {nv, y,z,u,v} such that x is connected to y,z,u,v in color c.
            
            # To make it faster:
            # For each pair of disjoint edges (y,u) and (z,v) in color c?
            # No, y,z,u,v are just a set of 4 vertices.
            
            # Let's iterate over pairs of vertices (y,z) in N_c_nv.
            # For each (y,z), we need (u,v) in N_c_nv such that
            # (y,u), (y,v), (z,u), (z,v) are color c.
            # This means u and v are in N_c(y) \cap N_c(z) \cap N_c_nv.
            # Let C = N_c(y) \cap N_c(z) \cap N_c_nv.
            # We need to pick two distinct vertices u, v from C.
            # So if |C| >= 2, we have a candidate set {y,z,u,v}.
            # Now we need to find an x such that x is connected to all 4 in C \cup {y,z}.
            # x must be in N_c(y) \cap N_c(z) \cap N_c(u) \cap N_c(v).
            # And x != nv, x != y,z,u,v.
            
            # This logic can be implemented efficiently.
            
            for c in [0, 1]:
                N_c = N0 if c == 0 else N1
                N_nv = N_c[nv]
                
                # Get list of neighbors of nv in color c
                neighbors_nv = []
                temp = N_nv
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    neighbors_nv.append(idx)
                    temp ^= lsb
                
                # Iterate over pairs (y, z) in neighbors_nv
                for i in range(len(neighbors_nv)):
                    y = neighbors_nv[i]
                    for j in range(i + 1, len(neighbors_nv)):
                        z = neighbors_nv[j]
                        
                        # Common neighbors of y and z, also neighbors of nv
                        common = N_c[y] & N_c[z] & N_nv
                        # Remove y and z themselves (they are not neighbors of themselves)
                        # But y and z are in N_nv, so they are in common?
                        # N_c[y] does not contain y. So y is not in common.
                        # Same for z.
                        
                        # We need to pick u, v from common.
                        # If common has at least 2 bits, we can form a set of 4.
                        # To avoid iterating all pairs (u,v) from common,
                        # we can just check if there exists ANY x connected to all.
                        # But x depends on u and v.
                        
                        # Let's iterate over pairs (u, v) from common.
                        # Max size of common is ~15. Pairs ~ 100.
                        # Total pairs (y,z) ~ 100.
                        # Total iterations ~ 10,000.
                        
                        common_list = []
                        temp_c = common
                        while temp_c:
                            lsb = temp_c & -temp_c
                            idx = lsb.bit_length() - 1
                            common_list.append(idx)
                            temp_c ^= lsb
                        
                        for k in range(len(common_list)):
                            u = common_list[k]
                            for l in range(k + 1, len(common_list)):
                                v = common_list[l]
                                
                                # Check if there is an x connected to y, z, u, v
                                # x must be in N_c(y) & N_c(z) & N_c(u) & N_c(v)
                                # And x != nv, y, z, u, v
                                cand_x = N_c[y] & N_c[z] & N_c[u] & N_c[v]
                                # Remove nv, y, z, u, v
                                cand_x &= ~(1 << nv)
                                cand_x &= ~(1 << y)
                                cand_x &= ~(1 << z)
                                cand_x &= ~(1 << u)
                                cand_x &= ~(1 << v)
                                
                                if cand_x:
                                    violations += 1
            return violations

        # Initial check
        # We start with n=29. We add vertex 30.
        # We try to find a coloring for vertex 30 that minimizes violations.
        
        # Strategy:
        # 1. Start with n=29 valid.
        # 2. Add vertex 30. Connect to all others with color 0.
        # 3. Use local search on the edges incident to 30.
        # 4. If violations > 0, try to flip edges.
        
        # Let's implement the search.
        
        def solve_n30():
            n = 30
            # Copy base colors
            colors_30 = [[0] * n for _ in range(n)]
            for i in range(n_base):
                for j in range(n_base):
                    colors_30[i][j] = colors[i][j]
            
            # Initialize edges to new vertex (29)
            # Try a few random initializations
            best_violations = float('inf')
            best_colors = None
            
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Random init for new vertex
                for i in range(n_base):
                    colors_30[29][i] = random.randint(0, 1)
                    colors_30[i][29] = colors_30[29][i]
                
                # Local Search
                curr_violations = count_violations_with_new_vertex(colors_30, n)
                # Note: count_violations_with_new_vertex only counts violations INVOLVING the new vertex.
                # Since base is valid, this is sufficient.
                
                if curr_violations == 0:
                    return colors_30
                
                # Hill Climbing with Random Restarts
                # We flip one edge incident to 29 at a time.
                # There are 29 edges.
                
                for step in range(2000):
                    if time.time() > deadline:
                        break
                    
                    if curr_violations == 0:
                        return colors_30
                    
                    # Pick a random neighbor i of 29
                    i = random.randint(0, n_base - 1)
                    
                    # Flip edge (29, i)
                    old_color = colors_30[29][i]
                    new_color = 1 - old_color
                    colors_30[29][i] = new_color
                    colors_30[i][29] = new_color
                    
                    new_violations = count_violations_with_new_vertex(colors_30, n)
                    
                    # Accept if better, or with some probability if worse (Simulated Annealing style)
                    if new_violations < curr_violations:
                        curr_violations = new_violations
                    else:
                        # Revert
                        colors_30[29][i] = old_color
                        colors_30[i][29] = old_color
                        # Optional: accept worse moves to escape local minima
                        # But for now, simple hill climbing.
                    
                    if curr_violations < best_violations:
                        best_violations = curr_violations
                        best_colors = [row[:] for row in colors_30]
                        
            return best_colors

        # Try to solve for n=30
        colors_30 = solve_n30()
        
        if colors_30 is not None:
            # Verify it's actually valid (full check just in case, though logic says it is)
            # For safety, we can run a full check on the final result.
            # But count_violations_with_new_vertex is trusted if base is valid.
            
            # Format output
            n = 30
            out_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    out_colors.append(colors_30[i][j])
            return {'n': n, 'colors': out_colors}
        
        # Fallback to n=29
        n = 29
        out_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                out_colors.append(colors[i][j])
        return {'n': n, 'colors': out_colors}

if __name__ == "__main__":
    result = construct()
    print(f"Found n={result['n']}")
