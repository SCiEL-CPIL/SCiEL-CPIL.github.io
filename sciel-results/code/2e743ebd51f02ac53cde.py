import itertools

N = 40

# EVOLVE-BLOCK-START
def construct():
    # Define the set of edges
    # Base Hamiltonian Cycle: (0, 1), (1, 2), ..., (39, 0)
    base_edges = set()
    for i in range(N):
        j = (i + 1) % N
        base_edges.add((min(i, j), max(i, j)))
    
    # Chords from 0 to all even vertices 2, 4, ..., 38
    chord_targets = range(2, N, 2)
    chord_edges = set()
    for v in chord_targets:
        chord_edges.add((0, v))
        
    all_edges = sorted(base_edges | chord_edges)
    
    # Build the list of cycles for lengths 3 to 40
    cycles = []
    
    for length in range(3, N + 1):
        if length == N:
            # Full Hamiltonian cycle
            cyc = list(range(N))
        elif length % 2 == 1:
            # Odd length L.
            # Use chord (0, L-1).
            # Cycle: 0, 1, ..., L-1, 0
            # Vertices: 0 to L-1
            cyc = list(range(length))
        else:
            # Even length L.
            # Use chords (0, 2) and (0, L).
            # Cycle: 0, 2, 3, ..., L, 0
            # Vertices: 0, then 2 to L
            cyc = [0] + list(range(2, length + 1))
        
        cycles.append(cyc)
        
    return {'edges': all_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
