import time

N = 40

def construct():
    # Build C_40
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Add chords (0, k) for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
        
    edge_list = list(edges)
    
    # Build cycles
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            # Determine which chord to use
            # We need a cycle of 'length'
            # If length <= 21, use chord k = length - 1
            # Cycle: 0, 1, ..., k, 0
            # If length > 21, use chord k = 41 - length
            # Cycle: 0, 39, ..., k, 0
            
            if length <= 21:
                k = length - 1
                # Path 0 -> 1 -> ... -> k -> 0
                cyc = list(range(k + 1))
                # Ensure it's a cycle list [0, 1, ..., k]
                # The cycle is 0-1-...-k-0
                # Vertices: 0, 1, ..., k
                cycles.append(cyc)
            else:
                k = 41 - length
                # Path 0 -> 39 -> ... -> k -> 0
                # Vertices: 0, 39, 38, ..., k
                cyc = [0] + list(range(39, k - 1, -1))
                cycles.append(cyc)
                
    return {'edges': edge_list, 'cycles': cycles}
