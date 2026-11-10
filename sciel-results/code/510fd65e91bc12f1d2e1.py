import time

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Returns a dictionary with keys 'edges' and 'cycles'.
    'edges' is a list of undirected edge pairs (u, v) with 0 <= u < v < 40.
    'cycles' is a list of 38 cycles (lengths 3 to 40), each represented as a list of vertices.
    """
    N = 40
    start_time = time.time()
    # Safety margin: return early if we exceed 85 seconds
    # The construction is O(N), so this is unlikely to be triggered.
    
    # 1. Initialize the edge set with the Hamiltonian cycle (0-1-...-39-0)
    edges = set()
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Add chords from vertex 0 to vertices 2 through 20.
    # These chords allow forming cycles of all lengths 3 to 40.
    # Specifically, chord (0, k) creates cycles of length k+1 (path 0-1-...-k)
    # and length 41-k (path 0-39-...-k).
    # k ranges from 2 to 20.
    # Lengths covered:
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21, 21
    # This covers all lengths from 3 to 39. Length 40 is the Hamiltonian cycle.
    
    for k in range(2, 21):
        if time.time() - start_time > 85.0:
            # Fallback: if we are close to the limit, just return what we have.
            # This should not happen.
            break
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # 3. Prepare the edge list
    edge_list = list(edges)
    
    # 4. Construct the cycles
    cycles = []
    # We need cycles of lengths 3, 4, ..., 40
    for L in range(3, N + 1):
        if time.time() - start_time > 85.0:
            break
            
        if L == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        elif L <= 21:
            # Cycle: 0 -> 1 -> 2 -> ... -> L-1 -> 0
            # Requires chord (0, L-1).
            # Since L-1 <= 20, and L >= 3, L-1 >= 2.
            # Chord (0, L-1) is in the set.
            cyc = list(range(L))
        else:
            # L > 21
            # Cycle: 0 -> 39 -> 38 -> ... -> k -> 0
            # Length L = 41 - k  => k = 41 - L
            # Since L <= 39, k >= 2.
            # Since L >= 22, k <= 19.
            # Chord (0, k) is in the set (2..20).
            k = 41 - L
            # Vertices: 0, 39, 38, ..., k
            # Note: range(39, k-1, -1) generates 39, 38, ..., k
            cyc = [0] + list(range(39, k - 1, -1))
        
        cycles.append(cyc)

    # Ensure we have exactly N-2 cycles (38 cycles)
    # If the loop broke early, we might need to fill, but the logic is O(N)
    # and fast, so it shouldn't break.
    while len(cycles) < N - 2:
        # This block is for safety only
        L = len(cycles) + 3
        if L == N:
            cyc = list(range(N))
        elif L <= 21:
            cyc = list(range(L))
        else:
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycles.append(cyc)

    return {'edges': edge_list, 'cycles': cycles}
