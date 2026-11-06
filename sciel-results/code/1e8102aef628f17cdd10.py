def construct():
        N = 40
        edges = []
        # Base cycle
        for i in range(N):
            edges.append((i, (i+1) % N))
        
        # Chords from 0 to 2..20
        for i in range(2, 21):
            edges.append((0, i))
        
        cycles = []
        for L in range(3, N+1):
            if L <= 21:
                # Cycle: 0, 1, ..., L-1
                cycle = list(range(L))
            else:
                # Cycle: 0, N-1, ..., 41-L
                # i = 41 - L
                i = 41 - L
                # Vertices: 0, N-1, N-2, ..., i
                cycle = [0] + list(range(N-1, i-1, -1))
            cycles.append(cycle)
            
        return {'edges': edges, 'cycles': cycles}
