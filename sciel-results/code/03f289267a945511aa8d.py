    def has_c6(u, v, adj_masks):
        # Check if adding (u,v) creates a C6.
        # Path: u - d - c - b - a - v
        # d in N(u) \ {v}
        # a in N(v) \ {u}
        # c in N(d) \ {u, v, d}
        # b in N(c) \ {u, v, d, c}
        # Check if a in N(b) \ {u, v, c, b}
        
        # Optimization: Precompute N2? No, dynamic.
        
        mask_u = adj_masks[u] & ~(1 << v)
        mask_v = adj_masks[v] & ~(1 << u)
        
        # Iterate d in N(u)
        d_mask = mask_u
        while d_mask:
            d = (d_mask & -d_mask).bit_length() - 1
            d_mask &= d_mask - 1
            
            # c in N(d)
            c_mask = adj_masks[d] & ~(1 << u) & ~(1 << v) & ~(1 << d)
            while c_mask:
                c = (c_mask & -c_mask).bit_length() - 1
                c_mask &= c_mask - 1
                
                # b in N(c)
                b_mask = adj_masks[c] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << c)
                while b_mask:
                    b = (b_mask & -b_mask).bit_length() - 1
                    b_mask &= b_mask - 1
                    
                    # Check if any neighbor of b is in N(v)
                    # and is not u, v, c, b, d
                    # a in N(b) ∩ N(v)
                    # a != u, v, b, c, d
                    
                    # N(b) is adj_masks[b]
                    # N(v) is mask_v
                    # Intersection
                    potential_a = adj_masks[b] & mask_v
                    # Remove b, c, d, u, v
                    potential_a &= ~(1 << b) & ~(1 << c) & ~(1 << d) & ~(1 << u) & ~(1 << v)
                    
                    if potential_a:
                        return True
        
        return False
    