"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i,j), max(i,j)))
    
    # Add chords (0, j) for j = 2..20
    for j in range(2, 21):
        edges.add((0, j))
    
    edge_list = sorted(edges)
    
    cycles = []
    for k in range(3, N + 1):
        if k <= 21:
            # Cycle: 0, 1, 2, ..., k-1, 0
            cycle = list(range(k))
        else:
            # Cycle: 0, 39, 38, ..., (40-k+1), 0
            # Vertices: 0, 39, 38, ..., (N-k+1)
            start = N - k + 1  # smallest vertex in the "tail"
            cycle = [0] + list(range(N-1, start-1, -1))
        cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
