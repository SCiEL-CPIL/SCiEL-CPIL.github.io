def construct():
    N = 40
    
    # 1. Base Cycle Edges
    # Vertices 0..39 in a ring.
    # Edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # 2. Chords from Hub 0
    # Add edges (0, k) for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
        
    edge_list = sorted(list(edges))
    
    # 3. Generate Cycles
    # We need cycles of length L for L in 3..40
    cycles = []
    
    # Helper to get the cycle for a given length L
    # Strategy:
    # If L == 40, use the full cycle 0-1-...-39-0
    # If L < 40:
    #   We have chords (0, k) for k=2..20.
    #   Chord (0, k) allows cycle 0-1-...-k-0 (length k+1)
    #   And cycle 0-k-(k+1)-...-39-0 (length 40-k+1)
    
    # Map length L to the required chord k
    # If we use short arc: L = k + 1 => k = L - 1
    # If we use long arc: L = 41 - k => k = 41 - L
    
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            # Determine which chord to use
            # We want k in [2, 20]
            # Option 1: k = L - 1. Valid if 2 <= L-1 <= 20 => 3 <= L <= 21.
            # Option 2: k = 41 - L. Valid if 2 <= 41-L <= 20 => 21 <= L <= 39.
            
            if L <= 21:
                k = L - 1
                # Cycle is 0, 1, ..., k
                cycle_verts = list(range(k + 1))
            else:
                k = 41 - L
                # Cycle is 0, k, k+1, ..., 39
                # Note: The path is k -> k+1 -> ... -> 39 -> 0
                # Vertices: k, k+1, ..., 39, 0
                # Wait, the cycle must be a list of vertices.
                # Path: 0 -> k -> k+1 -> ... -> 39 -> 0
                # Vertices in order: 0, k, k+1, ..., 39
                cycle_verts = [0] + list(range(k, N))
                
        cycles.append(cycle_verts)
        
    return {'edges': edge_list, 'cycles': cycles}
