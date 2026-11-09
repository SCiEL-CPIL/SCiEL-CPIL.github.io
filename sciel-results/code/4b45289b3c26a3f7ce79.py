import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 40

    edges_set = set()
    for i in range(N):
        edges_set.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))

    distances = list(range(2, 21))
    for d in distances:
        edges_set.add((0, d))

    cycles = {}
    cycles[40] = list(range(N))

    for d in distances:
        fwd = list(range(0, d + 1))
        bwd = [0] + list(range(d, N))

        if len(fwd) >= 3 and len(fwd) <= 40 and len(fwd) not in cycles:
            cycles[len(fwd)] = fwd
        if len(bwd) >= 3 and len(bwd) <= 40 and len(bwd) not in cycles:
            cycles[len(bwd)] = bwd

    if 21 not in cycles:
        cycles[21] = [0] + list(range(1, 21))

    valid = True
    for k in range(3, N + 1):
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

    if time.time() > DEADLINE:
        # Fallback to a known valid but potentially suboptimal structure if we ran out of time (unlikely)
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = []
        for k in range(3, N + 1):
            cycles_fb.append(list(range(k)))
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}

    if valid:
        out = [cycles[k] for k in range(3, N + 1)]
        return {'edges': [list(e) for e in edges_set], 'cycles': out}
    else:
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = []
        for k in range(3, N + 1):
            cycles_fb.append(list(range(k)))
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}
