    def remove_edge(adj, u, v, common):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Reverse the updates
        nu_mask = adj[u]  # N(u) after removal (doesn't include v)
        mask = nu_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            common[v][w] -= 1
            common[w][v] -= 1
        
        nv_mask = adj[v]  # N(v) after removal (doesn't include u)
        mask = nv_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            common[u][w] -= 1
            common[w][u] -= 1
