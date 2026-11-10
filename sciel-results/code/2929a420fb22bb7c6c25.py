import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    # all_edges is list of (u, v)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Precompute neighbor lists for quick access? 
    # Bitmasks are better.
    
    def creates_q3(u, v, adj):
        # adj is a list of 16 ints, where adj[x] is bitmask of neighbors
        # Check if adding (u,v) creates a Q3.
        # We assume (u,v) is NOT yet in adj.
        
        # Precompute neighbor lists for u and v excluding each other
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Get neighbor lists as lists for iteration
        nu = []
        mask = mask_u
        while mask:
            lsb = mask & -mask
            x = lsb.bit_length() - 1
            nu.append(x)
            mask ^= lsb
        
        nv = []
        mask = mask_v
        while mask:
            lsb = mask & -mask
            x = lsb.bit_length() - 1
            nv.append(x)
            mask ^= lsb
        
        # Iterate over pairs a, b in nu (a < b)
        for i in range(len(nu)):
            a = nu[i]
            for j in range(i+1, len(nu)):
                b = nu[j]
                
                # c must be in N(v) ∩ N(a), excluding u, v, a, b
                # d must be in N(v) ∩ N(b), excluding u, v, a, b, c
                
                mask_ca = mask_v & adj[a] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                mask_db = mask_v & adj[b] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                
                # Iterate c in mask_ca
                mask_c = mask_ca
                while mask_c:
                    lsb_c = mask_c & -mask_c
                    c = lsb_c.bit_length() - 1
                    mask_c ^= lsb_c
                    
                    # d must be in mask_db, excluding c
                    mask_d = mask_db & ~(1 << c)
                    
                    # e must be in N(a) ∩ N(b), excluding u, v, a, b, c, d
                    # f must be in N(c) ∩ N(d), excluding u, v, a, b, c, d, e
                    # (e, f) must be an edge
                    
                    # Precompute mask for e candidates
                    used_mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c)
                    mask_e_cand = adj[a] & adj[b] & ~used_mask
                    
                    if mask_e_cand == 0:
                        continue
                    
                    # Iterate d in mask_d
                    mask_d_iter = mask_d
                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d
                        
                        # Now check if there exists e in mask_e_cand and f in N(c) ∩ N(d)
                        # such that (e, f) is an edge
                        # f must be in N(c) ∩ N(d) & ~(used_mask | (1 << d) | (1 << e))
                        
                        mask_f_base = adj[c] & adj[d] & ~(used_mask | lsb_d)
                        
                        if mask_f_base == 0:
                            continue
                        
                        # Check if any e in mask_e_cand has an edge to some f in mask_f_base
                        # This is equivalent to: for some e in mask_e_cand, adj[e] & mask_f_base != 0
                        # We can check: OR of adj[e] for e in mask_e_cand, intersected with mask_f_base
                        
                        # To speed up, iterate e
                        mask_e = mask_e_cand
                        found = False
                        while mask_e:
                            lsb_e = mask_e & -mask_e
                            e = lsb_e.bit_length() - 1
                            mask_e ^= lsb_e
                            
                            if adj[e] & mask_f_base:
                                found = True
                                break
                        
                        if found:
                            return True
        
        return False

    # Greedy with Local Search
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        # Greedy Phase
        for u, v in all_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
                
        # Local Search Phase
        # Try to add more edges?
        # The greedy phase already tried all edges in random order.
        # So current_edges is a maximal set?
        # Not necessarily. If edge A was rejected because of B, and B is still there, A might be addable if we remove B?
        # But we don't remove edges in greedy.
        # So it's a maximal graph with respect to the order.
        
        # To improve, we can try "swaps" or "kicks".
        # Kick: Remove a random edge, then try to add as many edges as possible.
        
        # Let's do a few kicks per restart.
        for _ in range(10): # 10 kicks
            if time.time() > deadline: break
            
            # Pick a random edge to remove
            if not current_edges: break
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            
            # Remove it
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(idx)
            
            # Now try to add edges greedily again?
            # Or just try to add the removed one and others?
            # Let's try to add random non-edges.
            # We need a list of non-edges.
            # Or just shuffle all_edges and try to add any that are missing.
            
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline: break
                # Check if (u,v) is already in graph
                if (adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            
    return [[u, v] for u, v in best_edges]
