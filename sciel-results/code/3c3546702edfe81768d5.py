import time

N = 40
DEADLINE = 90  # seconds

def construct():
    start_time = time.time()
    
    # Check deadline immediately, though this function is O(1)
    if time.time() - start_time > DEADLINE - 1:
        return {'edges': [], 'cycles': []}

    edges = []
    # Base Hamiltonian cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from 0 to k for k in 2..20
    # This ensures cycles of all lengths 3..39
    # L=3..21 uses path 0..L-1 and chord (L-1, 0)
    # L=22..39 uses path 0..39..k and chord (k, 0) where k=41-L
    # Chords needed: (0,2)..(0,20)
    for k in range(2, 21):
        edges.append((0, k))
        
    # Remove duplicates just in case
    unique_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        unique_edges.add((u, v))
    
    edges_list = list(unique_edges)
    
    cycles = []
    # Lengths 3 to 40
    for L in range(3, N + 1):
        if L == 40:
            # Full Hamiltonian cycle
            cycle = list(range(N))
        elif 3 <= L <= 21:
            # Use chord (0, L-1)
            # Cycle: 0, 1, ..., L-1
            cycle = list(range(L))
        else:
            # L in 22..39
            # Use chord (0, k) where k = 41 - L
            k = 41 - L
            # Cycle: 0, 39, 38, ..., k
            # Vertices: 0, then 39 down to k
            cycle = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle)
        
    return {'edges': edges_list, 'cycles': cycles}
