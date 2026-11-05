import time

def construct():
    start_time = time.time()
    
    # Try n=17 first
    for n in [17, 16, 15, 14]:
        if time.time() - start_time > 115:
            break
            
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges
        
        # Symmetry breaking:
        # Fix colors of edges (0,1), (0,2), (0,3) to 0, 1, 2 respectively
        # This breaks color permutation symmetry
        # Index 0: (0,1)
        # Index 1: (0,2)
        # Index 2: (0,3)
        
        def set_edge(u, v, c, idx):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[idx] = c

        def unset_edge(u, v, c, idx):
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1

        # Apply symmetry breaking
        if n >= 4:
            set_edge(0, 1, 0, 0)
            set_edge(0, 2, 1, 1)
            set_edge(0, 3, 2, 2)
            start_idx = 3
        else:
            start_idx = 0

        def has_path_len_5(u, v, c):
            # Check if there is a simple path of length 5 from u to v in graph c
            # DFS
            # visited is a bitmask
            # stack: (node, depth, visited_mask)
            stack = [(u, 0, 1 << u)]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 5:
                    if node == v:
                        return True
                    continue
                
                # Get neighbors in color c
                neighbors = adj[c][node]
                # Iterate through set bits
                while neighbors:
                    # Extract lowest set bit
                    lsb = neighbors & -neighbors
                    nxt = lsb.bit_length() - 1
                    neighbors ^= lsb
                    
                    if visited & lsb:
                        continue
                    
                    # If we are at depth 4 and nxt is v, we found a path
                    if depth == 4 and nxt == v:
                        return True
                        
                    # Add to stack
                    stack.append((nxt, depth + 1, visited | lsb))
            return False

        def backtrack(idx):
            if time.time() - start_time > 110:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            # Try colors 0, 1, 2, 3
            # Heuristic: Try colors that are less constrained?
            # For now, 0, 1, 2, 3
            for c in range(4):
                # Check if adding (u,v) to color c creates a C6
                # This happens if there is a path of length 5 between u and v in G_c
                if has_path_len_5(u, v, c):
                    continue
                
                # Additional Pruning:
                # If we add (u,v) to color c, does it create a C6 with any OTHER edge?
                # No, C6 is formed by the new edge and 5 existing edges.
                # So the check above is sufficient.
                
                set_edge(u, v, c, idx)
                
                if backtrack(idx + 1):
                    return True
                
                unset_edge(u, v, c, idx)
            
            return False

        if backtrack(start_idx):
            return {'n': n, 'colors': colors}
            
    # Fallback: Return a valid small coloring if all else fails
    # The seed program provided a valid n=9. We can return that to ensure score > 0 if search fails.
    # But the search for 17 should succeed.
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
