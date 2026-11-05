import time

N = 40

def construct():
    # Structural change: Use a Hamiltonian cycle as the backbone.
    # Vertices 0..39 form a cycle: (0,1), (1,2), ..., (38,39), (39,0).
    # To form a cycle of length L (3 <= L <= 39), we use a chord (0, d)
    # where d is the distance along the cycle.
    # A chord (0, d) creates a cycle of length d+1 (forward) and 41-d (backward).
    # We need to cover all lengths 3..39.
    # This requires chords for distances d in {2, 3, ..., 20}.
    # (d=2 covers 3,39; d=3 covers 4,38; ... d=20 covers 21,21).
    # Length 40 is the Hamiltonian cycle itself.
    
    # 1. Build Hamiltonian Cycle Edges
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 2. Build Chords
    # Chords (0, d) for d in 2..20
    chord_edges = [(0, d) for d in range(2, 21)]
    
    edges = ham_edges + chord_edges
    
    # 3. Build Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == 40:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            if L <= 21:
                # Use forward path 0 -> 1 -> ... -> L-1 -> 0
                # Chord (0, L-1)
                cycle_verts = list(range(L))
            else:
                # Use backward path 0 -> 39 -> ... -> 41-L -> 0
                # Chord (0, 41-L)
                # Vertices: 0, 39, 38, ..., 41-L
                # Note: 41-L is the vertex index.
                # For L=22, 41-22=19. Vertices 0, 39, ..., 19.
                # For L=39, 41-39=2. Vertices 0, 39, ..., 2.
                end_vertex = 41 - L
                cycle_verts = [0] + list(range(N - 1, end_vertex - 1, -1))
            
            cycles.append(cycle_verts)
            
    return {'edges': edges, 'cycles': cycles}
