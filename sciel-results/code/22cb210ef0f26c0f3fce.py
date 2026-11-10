N = 40

def construct():
    # Base cycle C_N: edges (i, (i+1) % N) for i in 0..N-1
    edges = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))
    
    # Chords from vertex 0: pick S = {2, 3, ..., 20}
    S = set(range(2, 21))  # 2 through 20 inclusive
    for j in S:
        a, b = 0, j
        edges.add((min(a, b), max(a, b)))
    
    edge_list = sorted(edges)
    
    # Build cycles for each length L = 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full cycle
            cyc = list(range(N))
        else:
            if (L - 1) in S:
                # Cycle: 0, 1, 2, ..., L-1, 0 -> vertices [0, 1, ..., L-1]
                cyc = list(range(L))
            else:
                # Use chord (0, N-L+1), other direction
                j = N - L + 1  # this is in S
                # Cycle: 0, j, j+1, ..., N-1, 0
                cyc = [0] + list(range(j, N))
        cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
