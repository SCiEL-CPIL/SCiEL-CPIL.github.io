import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Edges
    edges = []
    # Hamiltonian cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from 0 to 2..20
    for k in range(2, 21):
        edges.append((0, k))
        
    # Remove duplicates if any (none in this construction)
    # Sort for consistency? Not required, but good.
    unique_edges = list(set(tuple(sorted(e)) for e in edges))
    
    # Cycles
    cycles = []
    
    # Length 3 to 21
    for L in range(3, 22):
        # Cycle 0-1-...-(L-1)-0
        cyc = list(range(L))
        cycles.append(cyc)
        
    # Length 22 to 39
    for L in range(22, 40):
        # Cycle 0-39-...-(41-L)-0
        # k = 41 - L
        k = 41 - L
        # Path from 39 down to k
        cyc = [0] + list(range(39, k - 1, -1))
        # Check: for L=22, k=19. range(39, 18, -1) -> 39..19.
        # Vertices: 0, 39, ..., 19.
        # Edges: (0,39), (39,38)...(20,19), (19,0).
        # 19 is connected to 0 via chord.
        # 39 is connected to 0 via cycle edge.
        cycles.append(cyc)
        
    # Length 40
    cycles.append(list(range(N)))
    
    # Verification (optional, but good for safety)
    # Check if all edges in cycles exist in graph
    edge_set = set(tuple(sorted(e)) for e in unique_edges)
    for cyc in cycles:
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i+1)%len(cyc)]
            if tuple(sorted((u,v))) not in edge_set:
                # Fallback to seed if invalid (should not happen)
                return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
                        'cycles': [list(range(k)) for k in range(3,N+1)]}
                
    return {'edges': unique_edges, 'cycles': cycles}
