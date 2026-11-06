def construct():
    N = 40
    # Base cycle: 0-1-2-...-39-0
    base_edges = [(i, i+1) for i in range(N-1)] + [(N-1, 0)]
    
    # Chords from 0 to d for d in 2..19
    chords = [(0, d) for d in range(2, 20)]
    
    # Additional chord for length 39
    extra_chord = (1, 3)
    
    edges = base_edges + chords + [extra_chord]
    
    # Build cycles for each length 3..40
    cycles = []
    for k in range(3, N+1):
        if k == N:  # length 40
            cycles.append(list(range(N)))
        elif k <= 20:  # from chord (0, k-1), cycle 1: [0, 1, ..., k-1]
            cycles.append(list(range(k)))
        else:  # k > 20, from chord (0, 41-k), cycle 2: [0, 41-k, 41-k+1, ..., 39]
            d = 41 - k
            cycles.append([0] + list(range(d, N)))
    
    return {'edges': edges, 'cycles': cycles}
