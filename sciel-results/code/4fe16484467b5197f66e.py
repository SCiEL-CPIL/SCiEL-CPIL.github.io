import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    # 1. Construct the base cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # 2. Add chords from vertex 0 to vertices 2..20
    # This ensures cycles of all lengths 3..39 are present.
    # Chord (0, k) creates cycles of length k+1 and N-k+1.
    # We need to cover lengths 3..39.
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=20 -> 21, 21
    for k in range(2, N // 2 + 1):
        edges.add((0, k))
        
    if time.time() > deadline:
        return {'edges': list(edges), 'cycles': []} # Should not happen

    edge_list = list(edges)
    
    # 3. Construct cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            k = L - 1
            if k <= N // 2:
                # Cycle: 0 -> 1 -> ... -> k -> 0
                cycle_verts = list(range(k + 1))
            else:
                # Cycle: 0 -> N-k -> ... -> N-1 -> 0
                # Note: N+1-L is the start vertex for the "long" way
                start = N + 1 - L
                cycle_verts = [0] + list(range(start, N))
                
        cycles.append(cycle_verts)

    return {'edges': edge_list, 'cycles': cycles}

# EVOLVE-BLOCK-START
# Entry point: construct()
# EVOLVE-BLOCK-END
