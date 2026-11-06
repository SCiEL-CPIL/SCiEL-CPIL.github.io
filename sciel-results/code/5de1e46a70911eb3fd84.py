import time

def construct():
    t0 = time.time()
    deadline = 50.0
    
    N = 40
    
    # Base Hamiltonian cycle edges
    edges = set()
    for i in range(N):
        u = i
        v = (i + 1) % N
        edges.add((min(u, v), max(u, v)))
        
    # Chords
    # k in {2, ..., 19, 38}
    chord_ks = list(range(2, 20)) + [38]
    for k in chord_ks:
        u = 0
        v = k
        edges.add((min(u, v), max(u, v)))
        
    edge_list = list(edges)
    
    # Generate cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle = list(range(N))
        elif L == 39:
            # Forward cycle with k=38
            k = 38
            cycle = list(range(k + 1)) # 0..38
        elif L <= 20:
            # Forward cycle with k = L - 1
            k = L - 1
            cycle = list(range(k + 1)) # 0..k
        else:
            # Backward cycle with k = 40 - L
            k = N - L
            # Cycle: 0, k, k+1, ..., 39
            cycle = [0] + list(range(k, N))
            
        cycles.append(cycle)
        
    return {'edges': edge_list, 'cycles': cycles}
