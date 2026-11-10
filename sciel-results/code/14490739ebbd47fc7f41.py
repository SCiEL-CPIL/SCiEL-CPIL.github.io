def creates_q3_with_edge(adj, u, v):
    n_u = [i for i in range(16) if (adj[u] >> i) & 1]
    n_v = [i for i in range(16) if (adj[v] >> i) & 1]
    
    if len(n_u) < 3 or len(n_v) < 3:
        return False
        
    for w_rest in combinations(n_u, 3):
        # Construct mask_W
        mask_W = (1 << v)
        for w in w_rest:
            mask_W |= (1 << w)
            
        for u_rest in combinations(n_v, 3):
            # Construct mask_U
            mask_U = (1 << u)
            for x in u_rest:
                mask_U |= (1 << x)
                
            if mask_U & mask_W:
                continue
                
            # Check regularity
            # We know u is in U, v is in W.
            # Check u: neighbors in W must be 3.
            # Check v: neighbors in U must be 3.
            # Check others.
            
            ok = True
            # Check u
            if bin(adj[u] & mask_W).count('1') != 3:
                ok = False
            elif bin(adj[v] & mask_U).count('1') != 3:
                ok = False
            else:
                # Check other vertices in U
                for x in u_rest:
                    if bin(adj[x] & mask_W).count('1') != 3:
                        ok = False
                        break
                if ok:
                    for w in w_rest:
                        if bin(adj[w] & mask_U).count('1') != 3:
                            ok = False
                            break
            if ok:
                return True
    return False
