import time
from collections import deque

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # The fan graph: base cycle 0-1-2-...-39-0 plus chords (0,2), (0,3), ..., (0,20)
    # This gives 40 + 19 = 59 edges.
    # It is pancyclic: cycles of length L exist for all 3 <= L <= 40.
    # For L <= 20: cycle is 0, 1, 2, ..., L-1
    # For L > 20: cycle is 0, 39, 38, ..., 41-L
    # For L = 40: cycle is 0, 1, 2, ..., 39

    edges = [(i, (i + 1) % N) for i in range(N)]
    for k in range(2, 21):
        edges.append((0, k))

    # Build edge set for verification
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))

    # Generate cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == 40:
            cyc = list(range(N))
        elif L <= 20:
            cyc = list(range(L))
        else:
            # L is in 21..39
            # Cycle: 0, 39, 38, ..., 41-L
            # Length: 1 + (39 - (41-L) + 1) = 1 + (L - 1) = L
            # Vertices: 0, 39, 38, ..., 41-L
            k_start = 41 - L
            cyc = [0] + list(range(39, k_start - 1, -1))
        
        # Verify cycle validity
        # Check all vertices are distinct
        if len(set(cyc)) != L:
            raise ValueError(f"Cycle of length {L} has duplicate vertices: {cyc}")
        
        # Check all edges exist
        for i in range(L):
            a, b = cyc[i], cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                raise ValueError(f"Edge ({a}, {b}) missing in cycle of length {L}")
        
        cycles.append(cyc)

    # Verify no duplicate edges
    if len(edge_set) != len(edges):
        raise ValueError("Duplicate edges found")

    return {'edges': edges, 'cycles': cycles}
