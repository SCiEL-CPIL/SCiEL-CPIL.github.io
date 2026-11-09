import time
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    edges = set()
    cycles = []
    
    # 1. Base Hamiltonian Cycle
    # Edges (i, i+1) mod 40
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    if time.time() > deadline:
        # Fallback: return what we have (though this shouldn't happen for N=40)
        pass

    # 2. Chords from 0 to k for k in 2..20
    # This generates cycles of length k+1 and N-k+1
    for k in range(2, 21):
        edges.add((0, k))
        
    if time.time() > deadline:
        pass

    edge_list = list(edges)
    
    # Generate cycles for lengths 3..40
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        elif 3 <= L <= 21:
            # L = k + 1 => k = L - 1
            # Cycle: 0-1-...-(L-1)-0
            cycle = list(range(L))
        else:
            # 22 <= L <= 39
            # L = N - k + 1 => k = N + 1 - L
            # Cycle: 0-k-(k+1)-...-(N-1)-0
            k = N + 1 - L
            # Path from k to N-1, then 0
            # Vertices: 0, k, k+1, ..., N-1
            cycle = [0] + list(range(k, N))
            
        cycles.append(cycle)
        
        if time.time() > deadline:
            # If we run out of time, we might not have all cycles.
            # But for N=40, this loop is trivial.
            pass

    return {'edges': edge_list, 'cycles': cycles}
