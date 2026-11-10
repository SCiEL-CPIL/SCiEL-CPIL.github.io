def creates_q3_fast(adj, u, v):
    # adj is list of int masks
    # u, v are the new edge
    nu = adj[u] & ~(1 << v)
    nv = adj[v] & ~(1 << u)
    
    # Iterate p in N(u)
    p_mask = nu
    while p_mask:
        p_bit = p_mask & -p_mask
        p_mask -= p_bit
        p = p_bit.bit_length() - 1
        
        # Iterate q in N(u) \ {p}
        q_mask = nu & ~(1 << p)
        while q_mask:
            q_bit = q_mask & -q_mask
            q_mask -= q_bit
            q = q_bit.bit_length() - 1
            
            # Candidate r in N(v) \cap N(p)
            # Candidate s in N(v) \cap N(q)
            # Pairing 1: p-r, q-s
            r_mask = nv & adj[p] & ~(1 << p) & ~(1 << q)
            s_mask = nv & adj[q] & ~(1 << p) & ~(1 << q)
            
            if r_mask and s_mask:
                # We need to find r in r_mask, s in s_mask, r != s
                # such that there exist t, w
                # t in N(p) \cap N(q)
                # w in N(r) \cap N(s)
                # t != w, t,w not in {u,v,p,q,r,s}
                # and adj[t] & (1<<w)
                
                t_candidates = (adj[p] & adj[q]) & ~(1 << u) & ~(1 << v) & ~(1 << p) & ~(1 << q)
                if t_candidates:
                    r_iter = r_mask
                    while r_iter:
                        r_bit = r_iter & -r_iter
                        r_iter -= r_bit
                        r = r_bit.bit_length() - 1
                        
                        s_candidates = s_mask & ~(1 << r)
                        if not s_candidates: continue
                        
                        w_candidates_base = (adj[r] & adj[s]) if False else 0 # Need s first
                        # Wait, w depends on s.
                        
                        s_iter = s_candidates
                        while s_iter:
                            s_bit = s_iter & -s_iter
                            s_iter -= s_bit
                            s = s_bit.bit_length() - 1
                            
                            w_candidates = (adj[r] & adj[s]) & ~(1 << u) & ~(1 << v) & ~(1 << p) & ~(1 << q) & ~(1 << r) & ~(1 << s)
                            if not w_candidates: continue
                            
                            # Check if any t in t_candidates and w in w_candidates are connected
                            # t_candidates is fixed for p,q
                            # w_candidates depends on r,s
                            
                            # We need: exists t in t_c, w in w_c such that adj[t] & (1<<w)
                            # This is equivalent to: (union of neighbors of w_c) & t_c != 0?
                            # Or simpler: iterate smaller set?
                            
                            # Optimization: if t_candidates is small, iterate t.
                            # If w_candidates is small, iterate w.
                            
                            # Let's just check intersection of neighbors.
                            # For each w in w_candidates, if adj[w] & t_candidates: return True
                            
                            w_iter = w_candidates
                            while w_iter:
                                w_bit = w_iter & -w_iter
                                w_iter -= w_bit
                                w = w_bit.bit_length() - 1
                                if adj[w] & t_candidates:
                                    return True
            # Pairing 2: p-s, q-r
            r_mask2 = nv & adj[q] & ~(1 << p) & ~(1 << q)
            s_mask2 = nv & adj[p] & ~(1 << p) & ~(1 << q)
            if r_mask2 and s_mask2:
                t_candidates = (adj[p] & adj[q]) & ~(1 << u) & ~(1 << v) & ~(1 << p) & ~(1 << q)
                if t_candidates:
                    r_iter = r_mask2
                    while r_iter:
                        r_bit = r_iter & -r_iter
                        r_iter -= r_bit
                        r = r_bit.bit_length() - 1
                        s_candidates = s_mask2 & ~(1 << r)
                        if not s_candidates: continue
                        s_iter = s_candidates
                        while s_iter:
                            s_bit = s_iter & -s_iter
                            s_iter -= s_bit
                            s = s_bit.bit_length() - 1
                            w_candidates = (adj[r] & adj[s]) & ~(1 << u) & ~(1 << v) & ~(1 << p) & ~(1 << q) & ~(1 << r) & ~(1 << s)
                            if not w_candidates: continue
                            w_iter = w_candidates
                            while w_iter:
                                w_bit = w_iter & -w_iter
                                w_iter -= w_bit
                                w = w_bit.bit_length() - 1
                                if adj[w] & t_candidates:
                                    return True
    return False
