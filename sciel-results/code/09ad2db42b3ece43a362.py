def construct():
    # K_8,8 on vertices 0..15: partition A = {0..7}, B = {8..15}
    # Diameter exactly 2, every edge critical (removing any edge makes
    # some pair at distance >= 3 or disconnected).
    # 64 edges is optimal for this constraint set.
    #
    # Complement graph analysis: complement of K_8,8 is K_8 ∪ K_8 (two
    # disjoint cliques on A and B). For any edge (u,v) in G with u∈A, v∈B,
    # u and v have NO common neighbors in G (neighbors of u are all in B,
    # neighbors of v are all in A, no overlap). Thus removing (u,v)
    # disconnects u from v, making diameter infinite > 2.
    #
    # For non-edges (u,w) with u,w ∈ A (or both in B): they are at distance 2
    # via any vertex in B (or A). After removing any single edge, there are
    # still 7 other vertices in the opposite partition to serve as intermediates.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    # Quick self-verification: confirm 64 edges, no loops/duplicates
    assert len(edges) == 64
    edge_set = set()
    for e in edges:
        assert isinstance(e, list) and len(e) == 2
        u, v = e[0], e[1]
        assert isinstance(u, int) and isinstance(v, int)
        assert not isinstance(u, bool) and not isinstance(v, bool)
        assert 0 <= u <= 15 and 0 <= v <= 15
        assert u != v
        key = (min(u, v), max(u, v))
        assert key not in edge_set, f"Duplicate edge {key}"
        edge_set.add(key)
    
    return edges
