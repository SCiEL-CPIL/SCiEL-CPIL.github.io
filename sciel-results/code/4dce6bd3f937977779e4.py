"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # 1. Hamiltonian Cycle Edges
    hc_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 2. Chord Edges from 0 to k for k in 2..20
    # These chords, combined with the Hamiltonian cycle, generate all cycles of length 3..39
    chord_edges = [(0, k) for k in range(2, 21)]
    
    edges = hc_edges + chord_edges
    
    # Ensure edges are distinct (they are by construction, but good practice)
    # hc_edges contains (0,1) and (39,0).
    # chord_edges contains (0,2)..(0,20).
    # No overlap.
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        elif L <= 21:
            # Cycle 0-1-...-(L-1)-0
            # Uses chords (0, L-1) where L-1 is in 2..20
            # For L=3, k=2. Chord (0,2) exists.
            # For L=21, k=20. Chord (0,20) exists.
            cycles.append(list(range(L)))
        else:
            # L in 22..39
            # Cycle 0-39-...-k-0 where k = 41 - L
            # k is in 2..19
            k = 41 - L
            # Vertices: 0, 39, 38, ..., k
            cycle_vertices = [0] + list(range(39, k - 1, -1))
            cycles.append(cycle_vertices)
            
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
