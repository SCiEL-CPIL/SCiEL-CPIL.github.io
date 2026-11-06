N = 40

def construct():
    # Base 40-cycle: 0-1-2-...-39-0
    edges = [(i, (i+1) % N) for i in range(N)]
    
    # Chords from 0 to vertices 2,3,...,20
    for k in range(2, 21):
        edges.append((0, k))
    
    # Now construct cycles of all lengths 3-40
    cycles = []
    for length in range(3, N+1):
        if length == N:
            # Base cycle
            cycles.append(list(range(N)))
        else:
            # Find k such that k+1 == length or 41-k == length
            # k = length - 1 or k = 41 - length
            k = length - 1
            if k > 20:
                k = 41 - length
            # Cycle: 0, 1, 2, ..., k, 0 (length k+1)
            # or 0, k, k+1, ..., 39, 0 (length 41-k)
            if k + 1 == length:
                cycle = list(range(k+1))  # 0,1,...,k
            else:
                # 41-k == length, so k = 41-length
                # Cycle: 0, k, k+1, ..., 39, 0
                cycle = [0] + list(range(k, N))
            cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
