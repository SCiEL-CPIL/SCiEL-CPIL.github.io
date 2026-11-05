def _kautz_graph(d, k):
    """Generate Kautz graph K(d, k).
    Vertices are tuples of length k over 0..d-1 with no adjacent equal elements.
    Edges: v -> w if w = v[1:] + (y,) where y != v[-1].
    The graph is (d-1)-regular.
    Girth is 2k.
    """
    # Generate all valid vertices
    vertices = []
    def gen(prefix):
        if len(prefix) == k:
            vertices.append(tuple(prefix))
            return
        for i in range(d):
            if not prefix or prefix[-1] != i:
                gen(prefix + [i])
    gen([])
    
    # Map vertex to index
    idx = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    edges = []
    
    for v in vertices:
        i = idx[v]
        # Next vertex starts with v[1:]
        suffix = v[1:]
        last = v[-1]
        for y in range(d):
            if y == last:
                continue
            w = suffix + (y,)
            j = idx[w]
            # Add edge only once
            if i < j:
                edges.append((i, j))
            
    return n, edges
