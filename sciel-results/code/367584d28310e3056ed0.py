import time

def construct():
    start_time = time.time()
    N = 40

    edges = set()

    # Base cycle C_40
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # Fan chords from vertex 0 to vertices 2..20 (19 chords)
    for k in range(2, 21):
        edges.add((0, k))

    # Construct one vertex-list cycle for each length L = 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L <= 21:
            cyc = list(range(L))
        else:
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycles.append(cyc)

    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 85:
        # Return best result found (which is the complete solution)
        return {'edges': list(edges), 'cycles': cycles}

    return {'edges': list(edges), 'cycles': cycles}
