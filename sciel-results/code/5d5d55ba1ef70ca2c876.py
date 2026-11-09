"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = set()
    
    # Add Hamiltonian rim
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # Add chords from 0 to k for k in 2..20
    # This covers cycle lengths 3..39
    chords = list(range(2, 21))
    for k in chords:
        edges.add((0, k))
        
    # Convert to list of tuples for output
    edge_list = list(edges)
    
    cycles = []
    # Generate cycles for lengths 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            if L <= 21:
                # Use chord k = L - 1
                # Cycle: 0, 1, ..., L-1, 0
                k = L - 1
                cycle = list(range(0, L))
                cycles.append(cycle)
            else:
                # Use chord k = 41 - L
                # Cycle: 0, k, k+1, ..., 39, 0
                k = 41 - L
                # Vertices: k, k+1, ..., 39, 0
                # But cycle must start/end? No, just a list of vertices in order.
                # The cycle is 0 -> k -> k+1 -> ... -> 39 -> 0
                # List: [0, k, k+1, ..., 39]
                # Wait, the edge (39,0) is in the rim.
                # The edge (0,k) is the chord.
                # The edges (i, i+1) for i in k..38 are in the rim.
                # So the sequence of vertices is 0, k, k+1, ..., 39.
                # Length of this list is 1 + (39 - k + 1) = 41 - k = L.
                cycle = [0] + list(range(k, N))
                cycles.append(cycle)
                
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
