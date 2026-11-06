import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
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
    
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            if L <= 21:
                k = L - 1
                cycle_verts = list(range(k + 1))
            else:
                k = 41 - L
                cycle_verts = [0] + list(range(k, N))
                
        cycles.append(cycle_verts)
        
    # Deadline check - if we're close to timeout, return immediately
    if time.time() - start_time > DEADLINE:
        return {'edges': edge_list, 'cycles': cycles}
    
    return {'edges': edge_list, 'cycles': cycles}
