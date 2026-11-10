    def can_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        
        # Check pair (u, v) itself
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
            
        # Check all neighbors of v
        c = adj[v]
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            if x == u: continue
            # v is in N(x). Adding (u,v) makes v in N(u).
            # So v is common neighbor of u and x.
            # Current common neighbors of u and x: adj[u] & adj[x]
            cnt = (adj[u] & adj[x]).bit_count()
            if cnt > 1:
                return False
                
        # Check all neighbors of u
        c = adj[u]
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            if x == v: continue
            # u is in N(x). Adding (u,v) makes u in N(v).
            # So u is common neighbor of v and x.
            # Current common neighbors of v and x: adj[v] & adj[x]
            cnt = (adj[v] & adj[x]).bit_count()
            if cnt > 1:
                return False
                
        return True
