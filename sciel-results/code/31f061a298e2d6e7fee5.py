import time

N = 40

def construct():
    """
    Construct a sparse pancyclic graph on 40 vertices.
    
    Strategy: 
    - Base: Hamiltonian cycle (40 edges)
    - Add chords from vertex 0 to vertices 2..20 (19 chords)
    - Total edges: 59
    - Score: 59 - 40 = 19
    
    For each cycle length L in 3..40:
    - If L <= 21: use path 0-1-2-...-(L-1)-0, requiring chord (0, L-1)
    - If L >= 21: use path 0-(N-1)-(N-2)-...-k-0, requiring chord (0, k) where k = N - L + 1
    - L = 40: the Hamiltonian cycle itself
    """
    start_time = time.time()
    
    edges = set()
    
    # 1. Hamiltonian cycle: (i, (i+1) % N) for i in 0..39
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    # 2. Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    # 3. Generate cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle_verts = list(range(N))
        elif L <= 21:
            # Short path: 0, 1, 2, ..., L-1, back to 0
            # Requires chord (0, L-1)
            k = L - 1
            cycle_verts = list(range(k + 1))
        else:
            # Long path: 0, N-1, N-2, ..., k, back to 0
            # Requires chord (0, k) where k = N - L + 1
            # Length = (N - k) + 1 = L => k = N - L + 1
            k = N - L + 1
            cycle_verts = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle_verts)
    
    # Return the result
    return {
        'edges': [tuple(sorted(e)) for e in edges],
        'cycles': cycles
    }
