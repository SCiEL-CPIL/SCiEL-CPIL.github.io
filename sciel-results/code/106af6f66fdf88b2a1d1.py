N = 40

def construct():
    # 1. Construct the base cycle edges
    # Edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # 2. Add chords
    # We need chords k such that for all L in 3..N-1, 
    # either k=L-1 or k=N+1-L is in the set of chords.
    # For N=40 (even), we must include k=20.
    # For pairs {k, 40-k}, we pick k for k in 2..19.
    # So chords are (0, 2), (0, 3), ..., (0, 20).
    for k in range(2, N // 2 + 1):
        e = (0, k)
        if e not in edges:
            edges.add(e)
            
    # The edges set now contains all required edges.
    # Convert to list of tuples for the output format
    edge_list = list(edges)
    
    # 3. Construct cycles
    cycles = []
    
    # Cycle for length L
    # If L-1 <= N//2, use chord (0, L-1)
    # Else, use chord (0, N+1-L)
    
    for L in range(3, N + 1):
        if L == N:
            # Base cycle
            cycle_verts = list(range(N))
        else:
            k1 = L - 1
            k2 = N + 1 - L
            
            # Check which chord exists in our construction
            # We included chords 2..N//2
            if k1 <= N // 2:
                # Use chord (0, k1)
                # Cycle: 0 -> 1 -> ... -> k1 -> 0
                # Vertices: 0, 1, ..., k1
                cycle_verts = list(range(k1 + 1))
            else:
                # Use chord (0, k2)
                # Cycle: 0 -> k2 -> k2+1 -> ... -> N-1 -> 0
                # Vertices: 0, k2, k2+1, ..., N-1
                cycle_verts = [0] + list(range(k2, N))
        
        cycles.append(cycle_verts)
        
    return {'edges': edge_list, 'cycles': cycles}
