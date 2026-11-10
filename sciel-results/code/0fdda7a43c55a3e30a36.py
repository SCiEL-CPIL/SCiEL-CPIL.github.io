import time

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Returns a dict with 'edges' (list of undirected edge pairs) and 'cycles'
    (list of vertex-list cycles, one for each length 3..40 in that order).
    """
    t_start = time.time()
    deadline = t_start + 80.0  # 80s safety margin before 90s kill
    N = 40
    edges = set()

    # Add the Hamiltonian cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Add 19 strategic chords from vertex 0 to vertices 2..20
    # This gives us 40 + 19 = 59 edges total.
    for k in range(2, 21):
        if time.time() > deadline:
            break
        edges.add((0, k))

    # Build the edge list with consistent ordering (min, max)
    edge_list = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_list.append((u, v))

    # Build cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if time.time() > deadline:
            # Fallback: return what we have so far (should not happen in practice)
            break
        if L == N:
            # Full Hamiltonian cycle: 0, 1, 2, ..., 39
            cyc = list(range(N))
        elif L <= 21:
            # Cycle: 0 -> 1 -> 2 -> ... -> L-1 -> 0
            # Edges used: (0,1), (1,2), ..., (L-2,L-1), (L-1,0)
            # The chord (0, L-1) exists since 2 <= L-1 <= 20
            cyc = list(range(L))
        else:
            # Cycle: 0 -> 39 -> 38 -> ... -> k -> 0 where k = 41 - L
            # Vertices: 0, 39, 38, ..., k (total L vertices)
            # Edges used: (0,39), (39,38), ..., (k+1,k), (k,0)
            # The chord (0, k) exists since 2 <= k <= 19
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycles.append(cyc)

    # If we broke early due to deadline, pad cycles with valid ones
    # In practice this won't happen since the construction is O(N)
    while len(cycles) < N - 2:
        L = len(cycles) + 3
        if L <= 21:
            cyc = list(range(L))
        else:
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycles.append(cyc)

    return {'edges': edge_list, 'cycles': cycles}
