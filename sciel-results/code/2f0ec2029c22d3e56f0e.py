"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Base: Hamiltonian cycle 0-1-2-...-39-0 (40 edges)
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Add 19 chords from vertex 0 to vertices 2..20
    # Chord (0, j) creates cycles of length j+1 and 41-j
    for j in range(2, 21):  # j = 2, 3, ..., 20
        edges.append((0, j))
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Generate cycles for each length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Find j such that j+1 == k or 41-j == k
            # j+1 == k => j = k-1
            # 41-j == k => j = 41-k
            j1 = k - 1
            j2 = 41 - k
            # Use j1 if valid (2 <= j1 <= 20), else j2
            if 2 <= j1 <= 20:
                j = j1
            else:
                j = j2
            # Cycle: 0, 1, 2, ..., j (length j+1 = k) OR 0, j, j+1, ..., 39 (length 41-j = k)
            if j + 1 == k:
                # Use the short cycle: 0, 1, ..., j
                cycle = list(range(j + 1))
            else:
                # Use the long cycle: 0, j, j+1, ..., 39
                cycle = [0] + list(range(j, N))
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
