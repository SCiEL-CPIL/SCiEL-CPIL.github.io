N = 40

def construct():
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i,j), max(i,j)))
    
    # Chords from 0 to 2,3,...,20
    # This set of 19 chords allows constructing all cycles of length 3..40
    for d in range(2, 21):
        edges.add((0, d))
    
    edges = list(edges)
    
    cycles = []
    for L in range(3, N+1):
        if L <= 21:
            # Use short path: 0, 1, ..., L-1, 0
            # Requires chord (0, L-1) which is in [2, 20]
            cycle = list(range(0, L))
        else:
            # Use long path: 0, 39, ..., 41-L, 0
            # Requires chord (0, 41-L)
            # 41-L is in [1, 19]
            # If 41-L == 1, edge (0,1) is in Ham cycle
            # If 41-L in [2, 19], edge is in chords
            d = 41 - L
            cycle = [0] + list(range(39, d-1, -1))
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
