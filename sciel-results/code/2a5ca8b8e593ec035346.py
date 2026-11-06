<<<<<<< SEARCH
    def would_create_c6_fast(adj, u, v):
        """Fast bitset check: does adding edge (u,v) create a C6?
        A C6 using edge (u,v): u-a-b-c-d-v, all 6 distinct.
        Equivalent: there exist a in N(u)\\{v}, d in N(v)\\{u}, a!=d,
        and a path a-b-c-d of length 3 in G\\{u,v} with b,c not in {u,v,a,d}.
        We check: for each a in N(u)\\{v}, d in N(v)\\{u}, a!=d:
          Let S_a = N(a) \\ {u,v,d}
          Let T_d = N(d) \\ {u,v,a}
          We need b in S_a, c in T_d, edge(b,c) exists, b!=c.
          Equivalent: exists b in S_a such that (N(b) & T_d & ~{a}) != 0
          where we also need c != b (guaranteed since c in T_d and b not in T_d if b not neighbor of d... 
          actually b could be neighbor of d. Let's be careful.)
          
        Better: For fixed a, d:
          We need a path a-b-c-d of length 3 in the graph with u,v removed.
          That means: exists b, c such that:
            b in N(a) \\ {u,v,d}
            c in N(d) \\ {u,v,a}  
            edge(b,c) exists
            b != c
            b != a (already: b in N(a) and simple graph means b != a)
            c != d (already: c in N(d) and simple graph)
            b != d (ensured by b not in {d})
            c != a (ensured by c not in {a})
          
          So: for b in N(a)\\{u,v,d}, check if N(b) & (N(d)\\{u,v,a}) & ~(1<<b) is non-empty.
          But we also need c != b, which is handled by ~(1<<b).
          And we need c != a, handled by excluding a from N(d).
          
        Even faster: for fixed a,d:
          Let B = N(a) & ~(1<<u) & ~(1<<v) & ~(1<<d)
          Let C = N(d) & ~(1<<u) & ~(1<<v) & ~(1<<a)
          We need: exists b in B such that N(b) & C has a bit set that is not b.
          i.e., OR over b in B of (N(b) & C & ~(1<<b)) != 0
          
        We can precompute for each vertex b, N(b). Then for each (a,d) pair,
        we iterate over b in B and check N(b) & C & ~(1<<b).
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # Iterate over a in N(u)\{v}
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            # B = N(a) excluding u, v, and we'll exclude d per-d
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            mask_d = Nv & ~(1 << a)  # d != a
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                # B = N(a) \ {u,v,d}
                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                
                # C = N(d) \ {u,v,a}
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue
                
                # For each b in B, check if N(b) & C & ~(1<<b) != 0
                mask_b = B
                found = False
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    # Check if any neighbor of b is in C and is not b itself
                    if adj[b] & C & ~(1 << b):
                        found = True
                        break
                
                if found:
                    return True
        
        return False
=======
    def would_create_c6_fast(adj, u, v):
        """Optimized C6 check using bit operations.
        
        A C6 through edge (u,v): u-a-b-c-d-v where all vertices distinct.
        For each a in N(u)\{v}, d in N(v)\{u}, a!=d:
          Need path a-b-c-d in G-{u,v} with b,c not in {u,v,a,d}.
          Equiv: exists b in N(a)\{u,v,d} such that N(b) intersects N(d)\{u,v,a,b}.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue
                
                # Check if any b in B has a neighbor in C (excluding b itself)
                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    # N(b) & C should have a bit set that isn't b
                    if adj[b] & C & ~(1 << b):
                        return True
        
        return False
>>>>>>> REPLACE

