import time

N = 40

def construct():
    start_time = time.time()
    
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        
    # 16 chords from 0
    for k in range(2, 19): # 2 to 18
        edges.add((0, k))
        
    # 2 additional chords
    edges.add((1, 20))
    edges.add((1, 21))
    
    edge_list = sorted(list(edges))
    
    # We need to generate cycles.
    # We know the graph is pancyclic.
    # We need to provide a valid cycle for each length 3..40.
    
    cycles = {}
    
    # Length 40
    cycles[40] = list(range(N))
    
    # Lengths 3..19 from chords (0, k)
    for k in range(2, 19):
        L = k + 1
        cycles[L] = list(range(k+1))
        
    # Lengths 23..39 from chords (0, k)
    for k in range(2, 19):
        L = N - k + 1
        cycles[L] = [0] + list(range(N-1, k-1, -1))
        
    # Missing lengths: 20, 21, 22
    # From (1, 20):
    # Cycle 1: 1-2-...-20-1 (Length 20)
    cycles[20] = [1] + list(range(2, 21))
    # Cycle 2: 1-0-39-...-20-1 (Length 22)
    # Path 20 to 1 via 0: 20-21-...-39-0-1
    # Vertices: 1, 0, 39, ..., 20
    cycles[22] = [1, 0] + list(range(N-1, 19, -1)) # 39 down to 20
    
    # From (1, 21):
    # Cycle 1: 1-2-...-21-1 (Length 21)
    cycles[21] = [1] + list(range(2, 22))
    # Cycle 2: 1-0-39-...-21-1 (Length 22) - Already covered
    
    # Check if all 3..40 are in cycles
    final_cycles = []
    for L in range(3, N+1):
        final_cycles.append(cycles[L])
        
    return {'edges': edge_list, 'cycles': final_cycles}
