import time

N = 40

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Uses a base cycle C_40 and 19 chords from vertex 0 to vertices 2..20.
    This ensures cycles of all lengths 3..40.
    Total edges: 59.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    edges = set()
    
    # 1. Base Cycle C_40
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # 2. Chords (0, k) for k in 2..20
    # These chords allow cycles of length k+1 and 41-k
    # k=2 -> L=3, 39
    # k=3 -> L=4, 38
    # ...
    # k=19 -> L=20, 22
    # k=20 -> L=21
    for k in range(2, 21):
        edges.add((0, k))
        
    cycles = []
    
    # 3. Construct cycles for lengths 3..40
    for L in range(3, N + 1):
        if L == N:
            # Full cycle
            cyc = list(range(N))
        elif L <= 21:
            # Cycle using path 0..L-1 and chord (0, L-1)
            # Vertices: 0, 1, ..., L-1
            # Check: Edge (L-1, 0) exists?
            # If L-1 == 1, edge (1,0) is base cycle.
            # If L-1 >= 2, edge (0, L-1) is a chord.
            # If L-1 == 20, edge (0, 20) is a chord.
            # All covered.
            cyc = list(range(L))
        else:
            # Cycle using path k..39..0 and chord (0, k)
            # k = 41 - L
            # Vertices: 0, k, k+1, ..., 39
            # Order: 0 -> k -> k+1 -> ... -> 39 -> 0
            # Check: Edge (0, k) exists?
            # L > 21 => k < 20.
            # k ranges from 18 (L=23) down to 2 (L=39).
            # Chords are for k in 2..20.
            # So (0, k) exists.
            # Edge (39, 0) is base cycle.
            k = 41 - L
            cyc = [0] + list(range(k, N))
            
        cycles.append(cyc)
        
        # Time check
        if time.time() > deadline:
            break
            
    # If we broke early, we might have incomplete cycles, but the loop is fast.
    # For N=40, this is negligible.
    
    return {
        'edges': list(edges),
        'cycles': cycles
    }

# EVOLVE-BLOCK-END
