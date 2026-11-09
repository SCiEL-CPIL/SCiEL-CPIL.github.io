def construct():
    # Construct a graph consisting of 8 disjoint copies of K4.
    # This graph has n = 8 * 4 = 32 vertices.
    # Each K4 is a clique, so the independence number is 8 (one vertex from each K4).
    # Since 8 < 9, there is no independent set of size 9.
    # K4 does not contain a 5-cycle, and the disjoint union of such graphs does not either.
    # This satisfies all conditions and maximizes n among simple constructions.
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
