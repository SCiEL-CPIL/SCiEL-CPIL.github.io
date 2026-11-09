import time

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill

    # Build the edge set: 40 rim edges + 19 chords = 59 edges
    edges_set = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges_set.add((min(a, b), max(a, b)))

    # Chords from vertex 0 to vertices 2..20
    for d in range(2, 21):
        edges_set.add((0, d))

    # Build cycles for lengths 3..40
    cycles = {}
    # Hamiltonian cycle: the rim
    cycles[40] = list(range(N))

    # For each chord distance d (2..20), we get cycles of length d+1 and 41-d
    for d in range(2, 21):
        if time.time() > deadline:
            break
        # Forward path: 0, 1, 2, ..., d (length d+1)
        fwd = list(range(0, d + 1))
        if 3 <= len(fwd) <= 40 and len(fwd) not in cycles:
            cycles[len(fwd)] = fwd

        # Backward path: 0, d, d+1, ..., 39, 0 (length 41-d)
        bwd = [0] + list(range(d, N))
        if 3 <= len(bwd) <= 40 and len(bwd) not in cycles:
            cycles[len(bwd)] = bwd

    # Verify all cycles
    valid = True
    for k in range(3, N + 1):
        if time.time() > deadline:
            valid = False
            break
        if k not in cycles:
            valid = False
            break
        cyc = cycles[k]
        if len(cyc) != k:
            valid = False
            break
        if len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if (min(a, b), max(a, b)) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if valid:
        out = [cycles[k] for k in range(3, N + 1)]
        return {'edges': [list(e) for e in edges_set], 'cycles': out}
    else:
        # Fallback: fan graph (39+38=77 edges, cost 37)
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = []
        for k in range(3, N + 1):
            cycles_fb.append(list(range(k)))
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}
