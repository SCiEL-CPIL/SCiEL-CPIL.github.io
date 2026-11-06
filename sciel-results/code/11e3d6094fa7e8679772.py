import time

N = 40
DEADLINE_S = 85.0  # safety margin before 90s kill

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Uses a base cycle C_40 and 19 chords from vertex 0 to vertices 2..20.
    This ensures cycles of all lengths 3..40.
    Total edges = 40 (cycle) + 19 (chords) = 59.
    Score metric = edge_count - 40 = 19.
    """
    t0 = time.time()
    
    edges = set()
    
    # 1. Base Cycle C_40: (0,1), (1,2), ..., (39,0)
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # 2. Chords (0, k) for k in 2..20
    # These provide the "shortcuts" needed to form cycles of various lengths.
    for k in range(2, 21):
        edges.add((0, k))
        
    # Check time, though this construction is instant
    if time.time() - t0 > DEADLINE_S:
        # Fallback: just return what we have (which is already complete)
        pass

    cycles = []
    
    # 3. Construct cycles for lengths 3..40
    # Strategy:
    # For L in 3..21: Use the path 0-1-...-(L-1) and the chord (L-1, 0).
    #   This requires chord (0, L-1). Since L-1 is in 2..20 for L in 3..21, this works.
    #   Vertices: 0, 1, ..., L-1
    # For L in 22..39: Let k = 41 - L. Since L=22 -> k=19, L=39 -> k=2.
    #   We use the path 0-39-38-...-k and the chord (k, 0).
    #   This requires chord (0, k). Since k is in 2..19, this chord exists.
    #   Vertices: 0, k, k+1, ..., 39
    # For L=40: Use the full cycle 0-1-...-39-0.
    
    for L in range(3, N + 1):
        if L == N:
            # Full cycle
            cyc = list(range(N))
        elif L <= 21:
            # Cycle using path 0..L-1 and chord (0, L-1)
            # Vertices: 0, 1, ..., L-1
            cyc = list(range(L))
        else:
            # Cycle using path k..39..0 and chord (0, k)
            # k = 41 - L
            k = 41 - L
            # Vertices: 0, k, k+1, ..., 39
            # Order: 0 -> k -> k+1 -> ... -> 39 -> 0
            cyc = [0] + list(range(k, N))
            
        cycles.append(cyc)
        
    return {
        'edges': list(edges),
        'cycles': cycles
    }

# EVOLVE-BLOCK-END
