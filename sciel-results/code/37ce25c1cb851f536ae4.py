def construct():
    """C5-free graph on 32 vertices with independence number 8.
    
    Constructed as the disjoint union of 8 copies of K4.
    Each K4 is C5-free (no 5-cycle possible in 4 vertices).
    Independence number = 8 (one vertex from each K4).
    """
    n = 32
    edges = []
    for block in range(8):
        base = 4 * block
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
