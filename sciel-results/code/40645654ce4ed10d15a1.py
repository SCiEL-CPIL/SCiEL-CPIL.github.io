"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Hamiltonian cycle 0-1-...-39-0
    edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords from 0 to cover cycles 3..39
    # S = {2, ..., 20}
    # Chords (0, s) for s in S
    chords = [(0, s) for s in range(2, 21)]
    edges.extend(chords)
    
    # Generate cycles
    cycles = []
    # Length 40 is the Hamiltonian cycle
    cycles.append(list(range(N)))
    
    # We need cycles of length 3 to 39
    # For each k in 3..39, find a cycle
    # We constructed S such that for every k in 3..39, 
    # there is s in S such that k = s+1 or k = 41-s
    
    # Map length k to chord s
    # If k <= 20, use s = k-1
    # If k > 20, use s = 41-k
    
    for k in range(3, N):
        if k <= 20:
            s = k - 1
            # Cycle: 0 -> 1 -> ... -> s -> 0
            # Path 0 to s is 0, 1, ..., s
            # Edges: (0,1)...(s-1,s) and (s,0)
            # Length s+1 = k
            cyc = list(range(s + 1)) # 0, 1, ..., s
            # Ensure it's a cycle list
            # The edge (s, 0) exists
        else:
            s = 41 - k
            # Cycle: 0 -> 39 -> ... -> s -> 0
            # Path 0 to s going backwards
            # 0, 39, 38, ..., s
            # Length 40 - s + 1 = 41 - s = k
            cyc = [0] + list(range(N - 1, s - 1, -1))
            
        cycles.append(cyc)
        
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
