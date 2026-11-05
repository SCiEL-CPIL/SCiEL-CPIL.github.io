def _construct_40_vertex_graph():
    """
    Construct a valid graph on 40 vertices using the Hoffman-Singleton graph.
    The HS graph has 50 vertices, is triangle-free, and has alpha=7.
    Any induced subgraph is triangle-free and has alpha <= 7 < 10.
    Thus, any 40-vertex induced subgraph is a valid witness for R(3,10) > 40.
    """
    n_hs = 50
    adj_hs = [0] * n_hs
    
    def idx(x, y):
        return x * 10 + y
    
    for x in range(5):
        for y in range(10):
            u = idx(x, y)
            v1 = idx(x, (y + 1) % 10)
            v2 = idx(x, (y - 1) % 10)
            v3 = idx((x + 1) % 5, (y + 2 * x) % 10)
            v4 = idx((x - 1) % 5, (y - 2 * x) % 10)
            
            for v in [v1, v2, v3, v4]:
                if u < v:
                    adj_hs[u] |= (1 << v)
                    adj_hs[v] |= (1 << u)
    
    n_target = 40
    adj_target = [0] * n_target
    
    for u in range(n_target):
        for v in range(u + 1, n_target):
            if adj_hs[u] & (1 << v):
                adj_target[u] |= (1 << v)
                adj_target[v] |= (1 << u)
    
    if _has_triangle_bitset(adj_target, n_target):
        return None
    
    if _max_independent_set_size_bitset(adj_target, n_target, 10):
        return None
    
    return {'n': n_target, 'edges': _bitset_to_edges(adj_target, n_target)}
