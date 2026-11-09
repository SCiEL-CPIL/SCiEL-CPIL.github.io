N = 40

def construct():
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i,j), max(i,j)))
    
    # Chords from 0 to 2,3,...,19
    # d=2 gives lengths 3, 38
    # d=19 gives lengths 20, 21
    # These 18 chords cover all pairs {L, 41-L} for L in 3..20
    for d in range(2, 20):
        edges.add((0, d))
    
    edges = list(edges)
    
    cycles = []
    for L in range(3, N+1):
        if L <= 20:
            # L = d+1, d = L-1
            d = L - 1
            # Cycle 0, 1, ..., d
            cycle = list(range(0, d+1))
        elif L == 21:
            # L=21 corresponds to d=19 or d=20. 
            # d=19 gives 20, 21. 
            # Cycle for 21 using d=19: 0, 39, 38, ..., 19
            d = 19
            cycle = [0] + list(range(39, d-1, -1))
        else:
            # L > 21
            # L = 41-d, d = 41-L
            # Since L is in 22..39, d is in 19..2
            # d = 41 - L
            d = 41 - L
            # Cycle 0, 39, 38, ..., d
            cycle = [0] + list(range(39, d-1, -1))
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
