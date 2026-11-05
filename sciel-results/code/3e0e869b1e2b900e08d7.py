   def check_add(adj, u, v):
       # Returns (is_valid, type, culprit)
       # type: None, 'direct', 'sec_u', 'sec_v'
       
       # 1. Direct: N(u) & N(v)
       c = adj[u] & adj[v]
       if c.bit_count() > 2:
           # Find a culprit z in c
           # Just pick the lowest set bit for determinism or random?
           # Random is better for search diversity.
           # But we need to be fast.
           # We can just return the mask and let the caller pick?
           # Or pick one here.
           # For speed, just return the mask?
           # No, let's pick one.
           z = (c & -c).bit_length() - 1 # Lowest set bit
           return False, 'direct', z
           
       # 2. Secondary for u: For z in N(v), |N(u) & N(z)| <= 1
       # Note: This check is needed because adding (u,v) makes v a common neighbor of u and z.
       # So if |N(u) & N(z)| was already 2, it becomes 3.
       mask_v = adj[v]
       while mask_v:
           lsb = mask_v & -mask_v
           z = lsb.bit_length() - 1
           if (adj[u] & adj[z]).bit_count() > 1:
               return False, 'sec_v', z # z is in N(v), problematic with u
           mask_v ^= lsb
           
       # 3. Secondary for v: For z in N(u), |N(v) & N(z)| <= 1
       mask_u = adj[u]
       while mask_u:
           lsb = mask_u & -mask_u
           z = lsb.bit_length() - 1
           if (adj[v] & adj[z]).bit_count() > 1:
               return False, 'sec_u', z # z is in N(u), problematic with v
           mask_u ^= lsb
           
       return True, None, None
   