"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i,j), max(i,j)))
    
    # Add chords (0, j) for j = 2..18 and j = 20
    # This set of 18 chords allows covering all cycle lengths 3..40.
    # Single chords cover:
    # j=2..18 -> lengths 3..19 and 39..23
    # j=20 -> length 21
    # Two chords (0,2) and (0,20) cover length 20 (dist 18 + 2)
    for j in list(range(2, 19)) + [20]:
        edges.add((0, j))
        
    edge_list = sorted(edges)
    
    cycles = []
    for k in range(3, N + 1):
        if k == 40:
            cycle = list(range(N))
        elif k == 21:
            # From chord (0, 20)
            # Cycle: 0, 1, ..., 20, 0
            cycle = list(range(21))
        elif k == 20:
            # From chords (0, 2) and (0, 20)
            # Cycle: 0, 2, 3, ..., 20, 0
            # Length: 1 (0-2) + 18 (2-20) + 1 (20-0) = 20
            cycle = [0, 2] + list(range(3, 21))
        elif k <= 19:
            # From chord (0, k-1)
            # Cycle: 0, 1, ..., k-1, 0
            cycle = list(range(k))
        else:
            # k in 22..39
            # From chord (0, 40-k+1)
            # Let j = 40 - k + 1
            # Cycle: 0, 39, 38, ..., j, 0
            j = 40 - k + 1
            cycle = [0] + list(range(N-1, j-1, -1))
        cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
