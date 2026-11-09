def greedy():
    adj = [base_adj[i] for i in range(n)]
    chosen = []
    while diameter(adj, n) > 4:
        best_edge = None
        best_score = None
        for (u,v) in candidates:
            if (adj[u]>>v)&1: continue  # already present
            if (adj[u] & adj[v]) != 0: continue  # triangle
            # compute new diameter if we add (u,v)
            new_adj = adj[:]
            new_adj[u] |= (1<<v)
            new_adj[v] |= (1<<u)
            d = diameter(new_adj, n)
            # score: prefer smaller diameter, then fewer bad pairs
            if best_score is None or d < best_score:
                best_score = d
                best_edge = (u,v)
        if best_edge is None:
            break  # no edge can be added (shouldn't happen)
        u,v = best_edge
        adj[u] |= (1<<v)
        adj[v] |= (1<<u)
        chosen.append((u,v))
    return chosen
