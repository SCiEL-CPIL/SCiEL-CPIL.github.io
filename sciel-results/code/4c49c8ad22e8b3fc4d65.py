import time

N = 40

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Uses a base cycle C_40 and 19 chords from vertex 0 to vertices 2..20.
    This ensures cycles of all lengths 3..40.
    """
    edges = set()
    
    # 1. Base Cycle C_40
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # 2. Chords (0, k) for k in 2..20
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
