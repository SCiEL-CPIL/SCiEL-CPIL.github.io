    others = [x for x in range(n) if x != u and x != v]
    for T in itertools.combinations(others, 6):
        verts = tuple(list(T) + [u, v])
        if _is_q3(adj, verts):
             return True
    