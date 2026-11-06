"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Vertices 0..39
    # Base Hamiltonian Cycle: (0,1), (1,2), ..., (39,0)
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from 0 to S
    S = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22]
    for k in S:
        edges.append((0, k))
        
    # Remove duplicates just in case (though logic prevents it)
    edges = list(set(tuple(sorted(e)) for e in edges))
    
    cycles = []
    # Length 3..40
    for L in range(3, N + 1):
        if L == N:
            # Full cycle
            cycles.append(list(range(N)))
        elif L == 20:
            # Use chords 20 and 2. Path 2->3->...->20
            # Cycle: 0-2-3-...-20-0
            cycles.append([0] + list(range(2, 21)))
        elif L == 22:
            # Use chords 22 and 2. Path 2->3->...->22
            # Cycle: 0-2-3-...->22-0
            cycles.append([0] + list(range(2, 23)))
        elif L <= 19:
            # Use chord k = L-1
            # Cycle: 0-1-...-(L-1)-0
            cycles.append(list(range(L)))
        elif L == 21:
            # Use chord k = 20
            # Cycle: 0-20-21-...-39-0
            cycles.append([0] + list(range(20, 40)))
        else:
            # L in 23..39
            # Use chord k = 41-L
            # Cycle: 0-k-(k+1)-...-39-0
            k = 41 - L
            cycles.append([0] + list(range(k, 40)))
            
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
