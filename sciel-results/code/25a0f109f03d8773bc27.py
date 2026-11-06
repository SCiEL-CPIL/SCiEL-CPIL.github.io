import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 59.0  # 1s margin before 60s limit

    # 1. Construct the base cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # 2. Add chords from vertex 0 to vertices 2..20
    # This ensures pancyclicity with minimum edges (59 total)
    for k in range(2, N // 2 + 1):
        edges.add((0, k))
        
    if time.time() > deadline:
        # Fallback (should not happen)
        return {'edges': list(edges), 'cycles': []}

    edge_list = list(edges)
    
    # 3. Construct cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            k1 = L - 1
            k2 = N + 1 - L
            if k1 <= N // 2:
                # Cycle: 0 -> 1 -> 2 -> ... -> k1 -> 0
                cycle_verts = list(range(k1 + 1))
            else:
                # Cycle: 0 -> k2 -> k2+1 -> ... -> N-1 -> 0
                cycle_verts = [0] + list(range(k2, N))
        cycles.append(cycle_verts)
        
    if time.time() > deadline:
        return {'edges': edge_list, 'cycles': cycles}

    return {'edges': edge_list, 'cycles': cycles}
