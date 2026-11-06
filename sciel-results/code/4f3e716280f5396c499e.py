import time

N = 40
START_TIME = time.time()
DEADLINE = 80  # safety margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def _edge_set(edge_list):
    s = set()
    for a, b in edge_list:
        s.add((a, b) if a < b else (b, a))
    return s

def _check_cycles_valid(edges, cycles):
    eset = _edge_set(edges)
    for cyc in cycles:
        L = len(cyc)
        if L < 3 or L > N:
            return False
        if len(set(cyc)) != L:
            return False
        for i in range(L):
            u = cyc[i]
            v = cyc[(i + 1) % L]
            e = (u, v) if u < v else (v, u)
            if e not in eset:
                return False
    return True

def _build_18chord():
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    for k in range(2, 19):  # (0,k) for k=2..18
        edges.add((0, k))
    edges.add((19, 38))
    edge_list = list(edges)

    cycles = [None] * (N - 2)  # index 0..37 -> lengths 3..40
    # length 40
    cycles[37] = list(range(N))
    # lengths 3..19 via (0, L-1)
    for L in range(3, 20):
        cycles[L - 3] = list(range(L))  # 0,1,...,L-1
    # length 20 via (19,38) short
    cycles[20 - 3] = list(range(19, 39))  # 19..38
    # length 21 multi-chord
    cycles[21 - 3] = [0] + list(range(2, 20)) + [38, 39]  # 0,2..19,38,39
    # length 22 via (19,38) long
    cycles[22 - 3] = list(range(19, -1, -1)) + [39, 38]  # 19,18..0,39,38
    # lengths 23..39 via (0, 41-L)
    for L in range(23, 40):
        k = 41 - L
        cycles[L - 3] = [0] + list(range(39, k - 1, -1))
    return edge_list, cycles

def _build_19chord():
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    for k in range(2, 21):  # (0,k) k=2..20
        edges.add((0, k))
    edge_list = list(edges)
    cycles = [None] * (N - 2)
    cycles[37] = list(range(N))
    for L in range(3, 40):
        if L <= 21:
            cycles[L - 3] = list(range(L))
        else:
            k = 41 - L
            cycles[L - 3] = [0] + list(range(39, k - 1, -1))
    return edge_list, cycles

def construct():
    # Try the 18-chord construction; verify; fall back to 19-chord.
    edge_list, cycles = _build_18chord()
    if _check_cycles_valid(edge_list, cycles):
        # also ensure all lengths 3..40 present
        lens = sorted(len(c) for c in cycles)
        if lens == list(range(3, N + 1)):
            return {'edges': edge_list, 'cycles': cycles}
    edge_list, cycles = _build_19chord()
    return {'edges': edge_list, 'cycles': cycles}
