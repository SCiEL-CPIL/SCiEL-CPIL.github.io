N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = set()
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Add chords from vertex 0 to even vertices 2, 4, ..., 38
    # This allows forming:
    # 1. Odd cycles L = 2k+1 via chord (0, 2k)
    # 2. Even cycles L = 2m via chords (0, 2) and (0, 2m)
    for v in range(2, N, 2):
        edges.add((0, v))
    
    edges_list = sorted(list(edges))
    
    # Build cycles for each length 3..40
    cycles = []
    for k in range(3, N+1):
        if k == N:
            cycle = list(range(N))
        elif k % 2 == 1:
            # Odd length k. Chord (0, k-1)
            # Cycle: 0-1-...-(k-1)-0
            cycle = list(range(k))
        else:
            # Even length k.
            # Use chords (0, 2) and (0, k)
            # Cycle: 0-2-3-...-k-0
            # Vertices: 0, 2, 3, ..., k
            cycle = [0] + list(range(2, k+1))
        cycles.append(cycle)
    
    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
