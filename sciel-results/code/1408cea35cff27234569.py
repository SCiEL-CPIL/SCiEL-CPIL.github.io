    def _has_k33_fast(adj_mask):
        n = len(adj_mask)
        # Precompute N(i) as masks
        # adj_mask[i] already is N(i)
        
        for a in range(n):
            for b in range(a+1, n):
                common_ab = adj_mask[a] & adj_mask[b]
                # If common_ab has < 2 bits, we can't form a triple with c?
                # Actually c must be in common_ab? No, c is just another vertex.
                # We need N(a) & N(b) & N(c) to have a triangle.
                # So c must be a neighbor of a and b? No.
                # c is just a vertex.
                # But if c is not in N(a) or N(b), then N(a)&N(b)&N(c) is empty?
                # No, N(a) is set of neighbors of a.
                # If c is not neighbor of a, then c is not in N(a).
                # The intersection is over the *neighbors*.
                # So we iterate c over all vertices?
                # Optimization: c must be such that N(c) intersects N(a) and N(b).
                # Actually, just iterate c from b+1 to n.
                
                for c in range(b+1, n):
                    common_abc = common_ab & adj_mask[c]
                    if common_abc == 0: continue
                    # Check if common_abc has >= 3 bits
                    # And if subgraph on common_abc has a triangle
                    
                    # Quick check: popcount
                    if (common_abc & (common_abc - 1)) == 0: continue # < 2 bits
                    # Actually need >= 3.
                    # Let's check triangle existence in common_abc
                    
                    # Iterate x in common_abc
                    x_mask = common_abc
                    has_triangle = False
                    while x_mask:
                        # extract lowest bit
                        lsb = x_mask & (-x_mask)
                        x = lsb.bit_length() - 1
                        x_mask ^= lsb
                        
                        # Neighbors of x within common_abc
                        neighbors_x_in_common = adj_mask[x] & common_abc
                        # Remove x itself (not in adj_mask[x] anyway for simple graph)
                        # Check if neighbors_x_in_common has an edge
                        # i.e. exists y, z in neighbors_x_in_common such that adj[y] & adj[z] != 0
                        # Simpler: for each y in neighbors_x_in_common, if (adj_mask[y] & neighbors_x_in_common) != 0: return True
                        # Note: adj_mask[y] & neighbors_x_in_common will include y? No, y is not in adj_mask[y].
                        # But it will include other nodes.
                        
                        y_mask = neighbors_x_in_common
                        while y_mask:
                            lsb_y = y_mask & (-y_mask)
                            y = lsb_y.bit_length() - 1
                            y_mask ^= lsb_y
                            
                            # Check if y has any neighbor in neighbors_x_in_common
                            # adj_mask[y] & neighbors_x_in_common
                            if adj_mask[y] & neighbors_x_in_common:
                                return True
                    # End while x
        return False
    