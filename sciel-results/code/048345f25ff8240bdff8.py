import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin

    # Construction: 40-cycle + chords from vertex 0 to vertices 2..20
    # Edges:
    #   40-cycle: (0,1), (1,2), ..., (38,39), (39,0)
    #   Chords: (0,2), (0,3), ..., (0,20)
    # Total edges: 40 + 19 = 59
    # Score: 59 - 40 = 19

    edges = []
    # 40-cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    # Chords from 0 to 2..20
    for k in range(2, 21):
        edges.append((0, k))

    # Build set for fast lookup
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))

    # Generate cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L <= 21:
            # Cycle: [0, 1, 2, ..., L-1]
            # Edges: (0,1), (1,2), ..., (L-2, L-1), (L-1, 0)
            # (L-1, 0) is chord when L-1 >= 2, i.e., L >= 3
            cyc = list(range(L))
        elif L <= 39:
            # Cycle: [0, 39, 38, ..., 41-L]
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        else:  # L == 40
            cyc = list(range(N))

        # Verify all edges in cycle exist
        valid = True
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                valid = False
                break
        if not valid:
            # Fallback: try to find a valid cycle (should not happen)
            raise RuntimeError(f"Invalid cycle for length {L}: {cyc}")

        cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
