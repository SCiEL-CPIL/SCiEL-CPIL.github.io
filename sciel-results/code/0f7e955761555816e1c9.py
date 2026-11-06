N = 40

def construct():
    # 1. Construct the base cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # 2. Add chords from vertex 0 to vertices 2..20
    # For each L in 3..39, need chord (0, L-1) or (0, 41-L)
    # Pairs {k, 40-k} for k=2..19, plus k=20: total 19 chords
    for k in range(2, N // 2 + 1):
        edges.add((0, k))
    
    edge_list = list(edges)
    
    # 3. Construct cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            k1 = L - 1
            k2 = N + 1 - L
            if k1 <= N // 2:
                # Cycle: 0 -> 1 -> 2 -> ... -> k1 -> 0
                cycle_verts = list(range(k1 + 1))
            else:
                # Cycle: 0 -> k2 -> k2+1 -> ... -> N-1 -> 0
                cycle_verts = [0] + list(range(k2, N))
        cycles.append(cycle_verts)
    
    return {'edges': edge_list, 'cycles': cycles}

# EVOLVE-BLOCK-START
# Entry point: construct()
# EVOLVE-BLOCK-END
