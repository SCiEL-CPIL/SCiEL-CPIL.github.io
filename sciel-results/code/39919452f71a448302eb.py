def construct():
    N = 40
    
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords (0, k) for k in 2..19
    # These cover lengths {k+1, N-k}
    # k=2 -> {3, 39}
    # ...
    # k=19 -> {20, 21}
    chord_edges = [(0, k) for k in range(2, 20)]
    
    all_edges = ham_edges + chord_edges
    normalized_edges = []
    for u, v in all_edges:
        if u > v:
            u, v = v, u
        normalized_edges.append((u, v))
    unique_edges = list(set(normalized_edges))
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L <= 20:
            # Length L covered by k = L-1
            # Cycle: 0..L-1
            cyc = list(range(L))
        else:
            # Length L covered by N-k = L => k = N-L
            # Cycle: 0, k, k+1, ..., N-1
            k = N - L
            cyc = [0] + list(range(k, N))
        cycles.append(cyc)
        
    return {'edges': unique_edges, 'cycles': cycles}
