import time

N = 40

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Uses a base cycle C_40 and 19 chords from vertex 0 to vertices 2..20.
    This ensures cycles of all lengths 3..40.
    Total edges: 59.
    """
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s limit
    
    edges = set()
    
    # 1. Base Cycle C_40
    for i in range(N):
        edges.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
        
    # 2. Chords (0, k) for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
        
    cycles = []
    
    # 3. Construct cycles for lengths 3..40
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L <= 21:
            cyc = list(range(L))
        else:
            k = 41 - L
            cyc = [0] + list(range(k, N))
            
        cycles.append(cyc)
    
    # Deadline guard
    if time.time() > deadline:
        pass  # Still return the constructed result; it's deterministic and fast
    
    return {
        'edges': list(edges),
        'cycles': cycles
    }

# EVOLVE-BLOCK-END
