import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 40
    edges = set()
    
    # Base cycle C_N
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # Chords from 0: k in 2..20
    # Each chord (0,k) with cycle-distance k creates cycles of length k+1 and 41-k
    # k=2: lengths 3,39; k=3: 4,38; ...; k=19: 20,22; k=20: 21
    # This covers all lengths 3..39; length 40 is the base cycle
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list of tuples
    edge_list = list(edges)
    
    cycles = []
    # Lengths 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # The full cycle
            cycles.append(list(range(N)))
        elif L == 21:
            # k=20 case
            # Cycle 0-1-...-20-0
            cycles.append(list(range(21)))
        elif L <= 20:
            # k = L - 1
            # Cycle 0-1-...-k-0
            k = L - 1
            cycles.append(list(range(k + 1)))
        else:
            # L > 21
            # k = 41 - L
            # Cycle 0-k-(k+1)-...-39-0
            k = 41 - L
            # Vertices: 0, k, k+1, ..., 39
            cyc = [0] + list(range(k, N))
            cycles.append(cyc)
            
    return {'edges': edge_list, 'cycles': cycles}
