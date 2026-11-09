"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = []
    # Base cycle 0-1-...-39-0
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from 0 to i for i=2..20
    for i in range(2, 21):
        edges.append((0, i))
        
    # Remove duplicates just in case, though logic says distinct
    # (0,1) is in cycle. (0,2)..(0,20) are chords.
    # Cycle edges are (0,1), (1,2), ..., (39,0).
    # Chords are (0,2)..(0,20).
    # Are any chords already in cycle?
    # Cycle has (0,1) and (0,39).
    # Chords are (0,2)..(0,20). No overlap.
    
    unique_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        unique_edges.add((u, v))
    edges = list(unique_edges)

    cycles = []
    # We need cycles of length 3 to 40
    for k in range(3, N + 1):
        if k == N:
            # Full cycle
            cycles.append(list(range(N)))
        elif k <= 21:
            # Use chord i = k-1
            # Cycle: 0, 1, ..., k-1, 0
            # Vertices: 0, 1, ..., k-1
            # Edges: (0,1), (1,2), ..., (k-2, k-1), (k-1, 0)
            # Chord is (0, k-1).
            # This works for k=3 (i=2, chord 0-2).
            cycle_vertices = list(range(k))
            cycles.append(cycle_vertices)
        else:
            # k in 22..39
            # Use chord i = 41 - k
            # Cycle: 0, i, i+1, ..., 39, 0
            # Vertices: 0, i, i+1, ..., 39
            # Count: 1 + (39 - i + 1) = 41 - i = k.
            i = 41 - k
            cycle_vertices = [0] + list(range(i, N))
            cycles.append(cycle_vertices)
            
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
