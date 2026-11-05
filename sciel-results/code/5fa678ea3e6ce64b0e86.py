N = 40

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Uses a base cycle and 19 chords from vertex 0 to vertices 2..20.
    Total edges: 39 (base) + 19 (chords) = 58.
    This is likely the minimum edge count for a pancyclic graph on 40 vertices.
    """
    edges = set()
    
    # Base cycle: 0-1-2-...-39-0
    for i in range(N):
        edges.add(tuple(sorted((i, (i + 1) % N))))
    
    # Chords from 0 to k for k in 2..20
    # These chords allow forming cycles of all lengths 3..39
    # Cycle length L can be formed by chord (0, k) where L = k+1 or L = N-k+1
    # For k=2, lengths 3 and 39
    # For k=20, length 21
    for k in range(2, 21):
        edges.add(tuple(sorted((0, k))))
    
    # Convert to list of sorted tuples
    edge_list = sorted(edges)
    
    # Generate cycles of length 3 to 40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        elif L <= 21:
            # Use chord (0, L-1) and path 0-1-...-(L-1)-0
            # Vertices: 0, 1, ..., L-1
            # Edges: (0,1)..(L-2,L-1) base, (L-1,0) chord
            # Length: L-1 base edges + 1 chord = L
            # Wait, vertices are 0..L-1. Count is L.
            # Cycle: 0 -> 1 -> ... -> L-1 -> 0
            cyc = list(range(L))
        else:
            # 22 <= L <= 39
            # Use chord (0, k) where N - k + 1 = L => k = N - L + 1
            # Path: 0 -> N-1 -> ... -> k -> 0
            # Vertices: 0, N-1, N-2, ..., k
            # Count: 1 (for 0) + (N-1 - k + 1) = N - k
            # Wait, path 0-N-1-...-k has N-k edges?
            # Vertices involved: 0 and k..N-1.
            # Count: 1 + (N-1 - k + 1) = N - k.
            # We want L vertices. So N - k = L => k = N - L.
            # Let's re-verify the length formula.
            # Chord (0, k).
            # Path 1: 0-1-...-k. Length k+1.
            # Path 2: 0-(N-1)-...-k. Length N-k+1.
            # We need L = N - k + 1 => k = N - L + 1.
            # Example: L=39, N=40. k = 40-39+1 = 2.
            # Path 2: 0-39-...-2. Vertices: 0, 39, ..., 2.
            # Count: 1 + (39-2+1) = 39. Correct.
            k = N - L + 1
            # Cycle: 0, N-1, N-2, ..., k
            # Ensure distinct vertices and correct order
            cyc = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
