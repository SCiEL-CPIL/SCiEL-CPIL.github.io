import time

N = 40
_DEADLINE = 80  # seconds, safety margin before 90s kill

def construct():
    t0 = time.time()
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Add chords (0, k) for k = 2, 3, ..., 20
    # This covers cycle lengths:
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21
    # Total 19 chords.
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = sorted(edges)
    
    # Build cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if time.time() - t0 > _DEADLINE:
            # Safety: return best effort (should never trigger for this O(1) construction)
            break
        
        if L == 40:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        elif L <= 21:
            # Cycle of length L uses chord (0, L-1) and path 0-1-...-(L-1)
            k = L - 1
            # Vertices 0, 1, ..., k
            cycles.append(list(range(k + 1)))
        else:
            # Cycle of length L uses chord (0, 41-L) and path 0-39-...-(41-L)
            k = 41 - L
            # Vertices 0, 39, 38, ..., k
            # Note: The path is 0-39-38-...-k-0
            # The cycle is 0, 39, ..., k
            cycles.append([0] + list(range(39, k - 1, -1)))
            
    return {'edges': edge_list, 'cycles': cycles}
