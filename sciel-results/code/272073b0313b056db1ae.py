    def is_valid_add_fast(u, v):
        # 1. Check pair (u, v)
        # Common neighbors of u and v do not change when adding (u,v)
        # But we must ensure they were < 3 before.
        if bin(adj[u] & adj[v]).count('1') >= 3:
            return False
        
        # 2. Check pairs (u, w) and (v, w) for all w != u, v
        for w in range(n):
            if w == u or w == w == v: # typo in thought, w != u and w != v
                continue
            
            # Pair (u, w)
            # New N(u) = N(u) U {v}
            # Common(u, w) = (N(u) U {v}) n N(w)
            # = (N(u) n N(w)) U ({v} n N(w))
            # Size is |N(u) n N(w)| + (1 if v in N(w) else 0)
            common_uw = bin(adj[u] & adj[w]).count('1')
            if v in (adj[w]): # v is neighbor of w
                if common_uw + 1 >= 3:
                    return False
            else:
                if common_uw >= 3:
                    return False
            
            # Pair (v, w)
            # New N(v) = N(v) U {u}
            # Common(v, w) = (N(v) U {u}) n N(w)
            # = (N(v) n N(w)) U ({u} n N(w))
            common_vw = bin(adj[v] & adj[w]).count('1')
            if u in (adj[w]): # u is neighbor of w
                if common_vw + 1 >= 3:
                    return False
            else:
                if common_vw >= 3:
                    return False
                
        return True
