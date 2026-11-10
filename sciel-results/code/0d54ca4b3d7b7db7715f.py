   def creates_q3(u, v, adj_mask):
       # adj_mask is list of 16 ints
       # Check if adding (u,v) creates Q3
       # We assume (u,v) is NOT in adj_mask yet, but we treat it as present for the check?
       # No, we check if the OTHER edges exist.
       
       # Candidates for a1, a2: neighbors of u (excluding v)
       # Candidates for b1, b2: neighbors of v (excluding u)
       
       nu = adj_mask[u] & ~(1 << v)
       nv = adj_mask[v] & ~(1 << u)
       
       # Get list of vertices
       # It might be faster to iterate over bits
       
       # Precompute list of vertices in nu and nv
       # Since n=16, we can just loop 0..15
       
       nu_verts = [i for i in range(16) if (nu >> i) & 1]
       nv_verts = [i for i in range(16) if (nv >> i) & 1]
       
       for a1 in nu_verts:
           for a2 in nu_verts:
               if a1 == a2: continue
               for b1 in nv_verts:
                   if b1 in (a1, a2): continue
                   for b2 in nv_verts:
                       if b2 in (a1, a2, b1): continue
                       
                       # Check edges (a1,b1) and (a2,b2)
                       if not ((adj_mask[a1] >> b1) & 1): continue
                       if not ((adj_mask[a2] >> b2) & 1): continue
                       
                       # Find c1: neighbor of a1 and a2, not in {u,v,a1,a2,b1,b2}
                       # Find c2: neighbor of b1 and b2, not in {u,v,a1,a2,b1,b2,c1}
                       
                       forbidden = (1 << u) | (1 << v) | (1 << a1) | (1 << a2) | (1 << b1) | (1 << b2)
                       
                       # c1 candidates: neighbors of a1 AND neighbors of a2 AND NOT forbidden
                       c1_cands = adj_mask[a1] & adj_mask[a2] & ~forbidden
                       
                       # c2 candidates: neighbors of b1 AND neighbors of b2 AND NOT forbidden
                       c2_cands = adj_mask[b1] & adj_mask[b2] & ~forbidden
                       
                       # Iterate c1
                       c1_bits = c1_cands
                       while c1_bits:
                           lsb = c1_bits & -c1_bits
                           c1 = lsb.bit_length() - 1
                           c1_bits ^= lsb
                           
                           # c2 must be in c2_cands, not c1, and adjacent to c1
                           c2_cands_valid = c2_cands & ~(1 << c1)
                           if not c2_cands_valid: continue
                           
                           # Check if any c2 in c2_cands_valid is adjacent to c1
                           # c1's neighbors & c2_cands_valid
                           if adj_mask[c1] & c2_cands_valid:
                               return True
       
       return False
   