import time

N = 40

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s

    def elapsed():
        return time.time() - t0

    # ---- Deterministic 19-chord construction (score 19) ----
    # Hamiltonian cycle H: 0-1-...-39-0  (40 edges)
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    # Chords (0, x) for x in 2..20
    for x in range(2, 21):
        edges.add((0, x))

    # Build cycles for each length 3..40
    cycles = []
    for L in range(3, N+1):
        if L == N:
            cyc = list(range(N))  # 0..39, closes via (39,0)
        elif L <= 20:
            x = L - 1  # short arc 0..x
            cyc = list(range(x+1))  # 0,1,...,x
        else:
            x = 41 - L  # long arc 0,39,38,...,x
            cyc = [0] + list(range(N-1, x-1, -1))  # 0,39,...,x
        cycles.append(cyc)

    edge_list = [tuple(sorted(e)) for e in edges]

    # ---- Verify ----
    def valid():
        eset = set(edge_list)
        if len(eset) != len(edge_list):
            return False
        for i in range(N):
            for j in range(i+1, N):
                pass
        # check each cycle
        for cyc in cycles:
            if len(cyc) != len(set(cyc)):
                return False
            m = len(cyc)
            for i in range(m):
                a, b = cyc[i], cyc[(i+1) % m]
                if (min(a,b), max(a,b)) not in eset:
                    return False
        # check lengths are exactly 3..40 in order
        if [len(c) for c in cycles] != list(range(3, N+1)):
            return False
        return True

    if valid():
        return {'edges': edge_list, 'cycles': cycles}

    # ---- Fallback: seed (fan) ----
    if elapsed() > DEADLINE:
        return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
                'cycles': [list(range(k)) for k in range(3,N+1)]}
    return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
            'cycles': [list(range(k)) for k in range(3,N+1)]}
