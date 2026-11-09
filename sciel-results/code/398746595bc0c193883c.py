N = 40

def construct():
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # Chords from 0 to 2..20
    for d in range(2, 21):
        edges.add((0, d))
        
    edges = list(edges)
    
    cycles = []
    for L in range(3, N + 1):
        if L <= 21:
            # Cycle 0 -> 1 -> ... -> L-1 -> 0
            cycle = list(range(L))
        else:
            # Cycle 0 -> 39 -> ... -> d -> 0 where d = 41 - L
            d = 41 - L
            cycle = [0] + list(range(39, d - 1, -1))
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
