"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 9
    # Known n=9 solution
    n0 = 9
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    # Try to find solutions for n from 17 down to 10
    # If we find one, we return it.
    # We prioritize higher n.
    
    for target_n in range(17, 9, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a valid coloring for target_n
        solution = _solve_for_n(target_n, deadline)
        if solution is not None:
            best_n = target_n
            best_colors = solution
            break
        else:
            # If we failed for 17, try 16, etc.
            # The loop handles this.
            pass
            
    return {'n': best_n, 'colors': best_colors}

def _solve_for_n(n, deadline):
    """
    Attempt to find a 4-coloring of K_n with no monochromatic C6.
    Returns list of colors or None if failed.
    """
    # Initialize random coloring
    num_edges = n * (n - 1) // 2
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Precompute edge indices
    # We need to map (i, j) to index in colors list
    # idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Build adjacency bitmasks for each color
    # adj[c][i] is a bitmask of neighbors of i in color c
    adj = [[0] * n for _ in range(4)]
    
    def build_adj():
        for c in range(4):
            for i in range(n):
                adj[c][i] = 0
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
                
    build_adj()
    
    full_mask = (1 << n) - 1
    
    def find_c6():
        """
        Find a monochromatic C6.
        Returns (color, [v0, v1, v2, v3, v4, v5]) or None.
        """
        for c in range(4):
            adj_c = adj[c]
            # Check all edges (u, v)
            for u in range(n):
                # Only consider edges where u < v to avoid duplicates? 
                # The cycle check is symmetric, but we just need to find ONE.
                # We can iterate all pairs.
                mask_u = adj_c[u]
                # Iterate neighbors v of u
                v_bits = mask_u
                while v_bits:
                    v_bit = v_bits & -v_bits
                    v = v_bit.bit_length() - 1
                    v_bits ^= v_bit
                    
                    if v <= u: continue # Optional: avoid double checking, but not strictly necessary for correctness, just speed
                    
                    # We have edge (u, v) in color c.
                    # Check for path u-x-y-z-v of length 4 (5 edges total in cycle? No, u-v is 1 edge. Path u-x-y-z-v is 4 edges. Total 5 edges? No.
                    # Cycle: u-v-x-y-z-w-u? No, C6 is 6 vertices, 6 edges.
                    # If we fix edge (u, v), the rest of the cycle is a path of length 4 from u to v?
                    # Path: u-a-b-c-d-v. Edges: (u,a), (a,b), (b,c), (c,d), (d,v).
                    # This path + edge (v,u) forms a C6.
                    # So we need a path of length 4 from u to v in the graph G_c \ {(u,v)}?
                    # Actually, the path vertices must be distinct from u and v.
                    
                    W = full_mask & ~(1 << u) & ~(1 << v)
                    N_u = adj_c[u] & W
                    N_v = adj_c[v] & W
                    
                    # We need x in N_u, z in N_v, and a path x-y-z of length 2 in W.
                    # x-y-z means y is common neighbor of x and z in W.
                    
                    x_bits = N_u
                    while x_bits:
                        x_bit = x_bits & -x_bits
                        x = x_bit.bit_length() - 1
                        x_bits ^= x_bit
                        
                        N_x = adj_c[x] & W
                        
                        # We need z in N_v such that N_x and adj_c[z] intersect in W.
                        # Let I = N_x & W.
                        # We need z in N_v such that (I & adj_c[z]) != 0.
                        # And the intersection vertex y must be distinct from x and z.
                        # Since y is in N_x, y != x.
                        # Since y is in adj_c[z], y != z.
                        # So just check intersection.
                        
                        z_bits = N_v
                        while z_bits:
                            z_bit = z_bits & -z_bits
                            z = z_bit.bit_length() - 1
                            z_bits ^= z_bit
                            
                            # Check if there is a common neighbor y in W
                            if (N_x & adj_c[z] & W):
                                # Found C6: u, x, y, z, v ... wait, the cycle is u-x-y-z-v-u?
                                # Edges: (u,x), (x,y), (y,z), (z,v), (v,u).
                                # This is a 5-cycle?
                                # u-x (1), x-y (2), y-z (3), z-v (4), v-u (5).
                                # C6 requires 6 edges.
                                # My path length logic was off.
                                # C6: v0-v1-v2-v3-v4-v5-v0.
                                # Fix edge (v0, v1).
                                # Need path v1-v2-v3-v4-v5-v0 of length 4?
                                # Edges: (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0).
                                # Yes, path of length 4 from v1 to v0?
                                # Vertices in path: v1, v2, v3, v4, v5, v0.
                                # Internal vertices: v2, v3, v4, v5.
                                # So we need a path of length 4 from u to v?
                                # u-a-b-c-d-v.
                                # Edges: (u,a), (a,b), (b,c), (c,d), (d,v).
                                # This path has 5 edges.
                                # Plus (v,u) makes 6 edges.
                                # So we need path of length 4 (5 edges) from u to v?
                                # No, "path of length 4" usually means 4 edges.
                                # Path with 4 edges: u-a-b-c-v. (5 vertices).
                                # Cycle: u-a-b-c-v-u. (5 edges). C5.
                                
                                # So for C6, we need a path of length 4 (4 edges) from u to v?
                                # u-a-b-c-v.
                                # Cycle: u-a-b-c-v-u.
                                # Vertices: u, a, b, c, v. (5 vertices).
                                # This is C5.
                                
                                # For C6:
                                # u-a-b-c-d-v.
                                # Edges: (u,a), (a,b), (b,c), (c,d), (d,v).
                                # This is a path of length 4 (4 edges) from u to v?
                                # No, u to v via a,b,c,d is 4 edges?
                                # u->a (1), a->b (2), b->c (3), c->d (4), d->v (5).
                                # It is a path of length 5 (5 edges).
                                # Wait.
                                # C6 has 6 edges.
                                # Remove one edge (u,v). We have a path of length 5 (5 edges) from u to v.
                                # Vertices: u, v1, v2, v3, v4, v.
                                # So we need to check for a path of length 5 from u to v.
                                # My previous check was for path of length 2 (x-y-z) which combined with u-x and z-v gives u-x-y-z-v (4 edges).
                                # That forms a C5.
                                
                                # So I need to check for path of length 5.
                                # u-a-b-c-d-v.
                                # a in N_u, d in N_v.
                                # Path a-b-c-d of length 3?
                                # a-b (1), b-c (2), c-d (3).
                                # So we need path of length 3 from a to d.
                                
                                # This is getting too slow to implement recursively in the inner loop.
                                # Given time constraints, I will fallback to a simpler, slower check that is correct.
                                pass
        return None
        
    # The above logic is flawed for C6.
    # Let's use a simpler, correct but slower check for the search.
    # Since n <= 17, we can afford a slightly slower check if the search is good.
    
    # Re-implement find_c6 correctly using DFS/Backtracking for one color at a time.
    # This is O(4 * n * D^5) worst case, but with pruning it's fast.
    
    def _has_c6_simple(adj_c, n):
        # adj_c is list of bitmasks
        # Convert to list of sets for faster iteration? Or use bitmasks.
        # Let's use bitmasks.
        for u in range(n):
            # DFS from u
            # Path: [u]
            # We want to find a path of length 5 from u that ends at a neighbor of u.
            # To avoid duplicates, we can restrict vertices to > u?
            # No, that misses cycles where u is not the smallest.
            # But we just need to find ANY C6.
            # If a C6 exists, it has a smallest vertex.
            # So we can just check: for each u, is there a C6 where u is the smallest vertex?
            
            # C6 with smallest vertex u:
            # u - v1 - v2 - v3 - v4 - v5 - u
            # All v_i > u.
            
            N_u = adj_c[u] & ~((1 << (u+1)) - 1) # Neighbors > u
            
            # DFS
            def dfs(current, depth, path_mask, path_list):
                if depth == 5:
                    # Check if current is connected to u
                    if (adj_c[current] >> u) & 1:
                        return True
                    return False
                
                # Next vertices must be > u and not in path
                # Also, to ensure simple path, we track visited.
                # We also want to avoid going back to u until the end.
                
                # Candidates: neighbors of current, > u, not in path_mask
                # Note: We allow going to any neighbor > u.
                # But we must not visit u.
                
                candidates = adj_c[current] & ~((1 << (u+1)) - 1) & ~path_mask
                # Also, we should not go to u. But u is not in candidates because candidates > u.
                
                c_bits = candidates
                while c_bits:
                    c_bit = c_bits & -c_bits
                    c = c_bit.bit_length() - 1
                    c_bits ^= c_bit
                    
                    if dfs(c, depth + 1, path_mask | c_bit, path_list + [c]):
                        return True
                return False
            
            # Start DFS from neighbors of u
            s_bits = N_u
            while s_bits:
                s_bit = s_bits & -s_bits
                s = s_bit.bit_length() - 1
                s_bits ^= s_bit
                
                if dfs(s, 1, s_bit, [s]):
                    return True
        return False

    # This _has_c6_simple is correct.
    # Now, the search.
    
    max_steps = 20000
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            return None
            
        # Check if current coloring is valid
        is_valid = True
        for c in range(4):
            if _has_c6_simple(adj[c], n):
                is_valid = False
                break
        
        if is_valid:
            return colors
        
        # If not valid, pick a random edge and flip
        idx = random.randint(0, num_edges - 1)
        old_c = colors[idx]
        new_c = random.choice([c for c in range(4) if c != old_c])
        
        # Update colors
        colors[idx] = new_c
        
        # Update adj
        # Find u, v from idx
        # Inverse of idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        # This is hard to invert directly.
        # Let's precompute edge list.
        
        # Precompute edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                
        u, v = edges[idx]
        
        adj[old_c][u] &= ~(1 << v)
        adj[old_c][v] &= ~(1 << u)
        adj[new_c][u] |= (1 << v)
        adj[new_c][v] |= (1 << u)
        
        step += 1
        
    return None

# EVOLVE-BLOCK-END
