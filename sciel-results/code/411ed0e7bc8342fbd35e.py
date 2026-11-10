    def is_valid_incremental(adj, u, v):
        for t in triples_with[u]:
            a, b, c = t
            common = adj[a] & adj[b]
            if len(common) < 3: continue
            common = common & adj[c]
            if len(common) >= 3:
                return False
        for t in triples_with[v]:
            a, b, c = t
            common = adj[a] & adj[b]
            if len(common) < 3: continue
            common = common & adj[c]
            if len(common) >= 3:
                return False
        return True
