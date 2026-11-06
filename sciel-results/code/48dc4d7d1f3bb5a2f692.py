<<<<<<< SEARCH
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) would create a C6."""
        # A new C6 involving edge (u,v) must use this edge.
        # So we need a path of length 5 from u to v not using edge (u,v).
        # i.e., u-a-b-c-d-v where all 6 vertices distinct and a!=v, d!=u, etc.
        
        # Path: u-a-b-c-d-v
        # a in N(u), d in N(v)
        # a != v, d != u
        # b in N(a)\{u}, c in N(d)\{v}
        # b != d, c != a, b != v, c != u
        # b in N(c)\{a} (edge b-c exists)
        # All 6 distinct: u,a,b,c,d,v
        
        Nu = adj[u] & ~(1 << v)  # neighbors of u except v
        Nv = adj[v] & ~(1 << u)  # neighbors of v except u
        
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            if a == v:
                continue
            
            Na_excl_u = adj[a] & ~(1 << u)
            
            mask_d = Nv
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                if d == u or d == a:
                    continue
                
                Nd_excl_v = adj[d] & ~(1 << v)
                
                # b in Na_excl_u, c in Nd_excl_v
                # b != d, c != a, b != v, c != u
                # edge b-c exists
                # b != c
                
                # For each b in Na_excl_u (b != d, b != v):
                mask_b = Na_excl_u & ~(1 << d) & ~(1 << v)
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    if b == u or b == a or b == d or b == v:
                        continue
                    
                    # c in Nd_excl_u (c != a, c != u, c != b)
                    # c must be neighbor of b
                    Nc_candidates = Nd_excl_v & adj[b]
                    Nc_candidates &= ~(1 << a) & ~(1 << u) & ~(1 << b)
                    
                    if Nc_candidates:
                        return True
        
        return False
=======
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) would create a C6 using fast bitset ops."""
        # A new C6: u-a-b-c-d-v where all 6 distinct, edge (u,v) is new.
        # a in N(u)\{v}, d in N(v)\{u}
        # b in N(a)\{u}, c in N(d)\{v}
        # edge (b,c) exists, b!=c, b!=d, c!=a, b!=v, c!=u, b!=u, c!=v, b!=a, c!=d
        # Also a!=d (otherwise path is u-a-d-v which is length 3, not 5)
        
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            Na_excl = adj[a] & ~(1 << u)  # N(a) \ {u}
            
            # d in N(v)\{u, a}
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                Nd_excl = adj[d] & ~(1 << v)  # N(d) \ {v}
                
                # b in N(a)\{u, d, v}
                # c in N(d)\{v, a, u}
                # Need edge (b,c), b != c, b != a, c != d
                # b in N(a)\{u,d,v,a} but a not in N(a) anyway
                # c in N(d)\{v,a,u,d} but d not in N(d) anyway
                
                # candidates for b: N(a) \ {u, d, v}
                # candidates for c: N(d) \ {v, a, u}
                # Check if any b in B and c in C have edge (b,c)
                # i.e., exists b in B such that adj[b] & C is non-empty (excluding b itself)
                
                B = Na_excl & ~(1 << d) & ~(1 << v)
                C = Nd_excl & ~(1 << a) & ~(1 << u)
                
                # For each b in B, check if adj[b] intersects C (and b not in C)
                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    # c must be in C, c != b, edge (b,c) exists
                    # adj[b] & C gives candidates, but exclude b itself
                    if adj[b] & C & ~(1 << b):
                        return True
        
        return False
>>>>>>> REPLACE

