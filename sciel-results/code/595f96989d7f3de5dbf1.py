    def _has_c6_bitset(adj_bits, n):
        # adj_bits[i] is an int bitmask of neighbors of i
        for u in range(n):
            # Iterate over neighbors v of u
            # We want a cycle u-v-w1-w2-w3-w4-u
            # Fix u.
            # Iterate over v in adj[u]
            # We need a path v -> ... -> u of length 5.
            # This is equivalent to checking if u is in adj[adj[adj[adj[adj[v]]]]]
            # but with vertex constraints.
            
            # Simpler: iterate over all pairs (v, w) that are neighbors of u?
            # No, let's stick to the path enumeration but with bitsets for intersection checks.
            
            for v in range(n):
                if not (adj_bits[u] >> v) & 1: continue
                if v <= u: continue # Symmetry reduction
                
                # We need a path v -> w1 -> w2 -> w3 -> w4 -> u
                # w1, w2, w3, w4 distinct from u, v.
                
                # Candidates for w1: neighbors of v, excluding u, v
                cands_w1 = adj_bits[v] & ~(1 << u) & ~(1 << v)
                
                while cands_w1:
                    lsb = cands_w1 & -cands_w1
                    w1 = lsb.bit_length() - 1
                    cands_w1 ^= lsb
                    
                    # Candidates for w2: neighbors of w1, excluding u, v, w1
                    cands_w2 = adj_bits[w1] & ~(1 << u) & ~(1 << v) & ~(1 << w1)
                    
                    while cands_w2:
                        lsb2 = cands_w2 & -cands_w2
                        w2 = lsb2.bit_length() - 1
                        cands_w2 ^= lsb2
                        
                        # Candidates for w3: neighbors of w2, excluding u, v, w1, w2
                        cands_w3 = adj_bits[w2] & ~(1 << u) & ~(1 << v) & ~(1 << w1) & ~(1 << w2)
                        
                        while cands_w3:
                            lsb3 = cands_w3 & -cands_w3
                            w3 = lsb3.bit_length() - 1
                            cands_w3 ^= lsb3
                            
                            # Candidates for w4: neighbors of w3, excluding u, v, w1, w2, w3
                            cands_w4 = adj_bits[w3] & ~(1 << u) & ~(1 << v) & ~(1 << w1) & ~(1 << w2) & ~(1 << w3)
                            
                            # Check if any w4 in cands_w4 is a neighbor of u
                            # i.e., (cands_w4 & adj_bits[u]) != 0
                            if cands_w4 & adj_bits[u]:
                                return True
        return False
    