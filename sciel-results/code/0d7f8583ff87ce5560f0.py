def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Returns a dict with 'edges' (list of undirected edge pairs) and 'cycles'
    (list of vertex-list cycles, one for each length 3..40 in that order).
    """
    N = 40
    edges = set()

    # Add the Hamiltonian cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Add 19 strategic chords from vertex 0 to vertices 2..20
    # This gives us 40 + 19 = 59 edges total.
    for k in range(2, 21):
        edges.add((0, k))

    # Build the edge list with consistent ordering (min, max)
    edge_list = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_list.append((u, v))
    
    # Sort edges for deterministic output? Not required but good practice.
    edge_list.sort()

    # Build cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle: 0, 1, 2, ..., 39
            cyc = list(range(N))
        elif L <= 21:
            # Cycle: 0 -> 1 -> 2 -> ... -> L-1 -> 0
            cyc = list(range(L))
        else:
            # Cycle: 0 -> 39 -> 38 -> ... -> k -> 0 where k = 41 - L
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycles.append(cyc)

    return {'edges': edge_list, 'cycles': cycles}
