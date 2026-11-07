def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 75.0  # safety margin before 90s wall-clock kill

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # edge index lookup (canonical undirected)
    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(min(u, v), max(u, v))] = i

    def key(u, v):
        return (min(u, v), max(u, v))

    # Classify edges into: bottom cycle, top cycle, matching
    bot_edges = []  # bottom cycle edge indices (undirected)
    top_edges = []  # top cycle edge indices
    match_edges = []
    for i, (u, v) in enumerate(edges):
        a, b = min(u, v), max(u, v)
        if 0 <= a < m and 0 <= b < m:
            bot_edges.append(i)
        elif a >= m and b >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)

    # Build adjacency (for balance check)
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Precompute edge-to-vertex mapping for faster balance check
    # For each vertex, store list of (neighbor, edge_index)
    adj_with_edge = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj_with_edge[u].append((v, i))
        adj_with_edge[v].append((u, i))

    def is_balanced(sgn):
        pot = np.zeros(n, dtype=np.int8)
        pot[0] = 1
        dq = deque([0])
        while dq:
            u = dq.popleft()
            pu = pot[u]
            for v, ei in adj_with_edge[u]:
                s = sgn[ei]
                want = int(pu * s)
                if pot[v] == 0:
                    pot[v] = want
                    dq.append(v)
                elif pot[v] != want:
                    return False
        return True

    # Precompute base adjacency matrix
    A_base = np.zeros((n, n))
    for (u, v) in edges:
        A_base[u, v] = 1
        A_base[v, u] = 1

    def spectral_radius(sgn):
        A = A_base.copy()
        for (u, v) in edges:
            s = sgn[idx[key(u, v)]]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    best_signs = None
    best_radius = float('inf')

    def consider(sgn):
        nonlocal best_signs, best_radius
        if is_balanced(sgn):
            return  # disconnected lift -> invalid
        r = spectral_radius(sgn)
        if r < best_radius:
            best_radius = r
            best_signs = sgn.copy()

    # ---- Phase 1: model sweep over cycle signs (matching fixed to +1) ----
    # Represent bottom/top cycles as a sign array of length m each (edge j connects j - (j+1 mod m)).
    base = np.ones(E, dtype=np.int8)
    # candidate list of (bottom_signs(m,), top_signs(m,))
    cands = []
    cands.append((np.ones(m, dtype=np.int8), np.ones(m, dtype=np.int8)))
    cands.append((-np.ones(m, dtype=np.int8), -np.ones(m, dtype=np.int8)))
    cands.append((-np.ones(m, dtype=np.int8), np.ones(m, dtype=np.int8)))
    cands.append((np.ones(m, dtype=np.int8), -np.ones(m, dtype=np.int8)))

    if 3 ** m <= 200000:
        arr = np.array(range(3 ** m), dtype=np.int32)
        tmp = arr
        for j in range(m):
            digits = np.empty(3 ** m, dtype=np.int8)
            tmp = np.where((tmp // (3 ** j)) % 3 == 0, -1,
                           np.where((tmp // (3 ** j)) % 3 == 1, 1, -1))
            # map 0->-1,1->1,2->-1
            digits = tmp
            if j == 0:
                b_digits = digits
            elif j == 1:
                t_digits = digits
        # simpler: build both arrays properly
    # Build full 3^m enumeration cleanly
    if 3 ** m <= 200000:
        b_digits = np.empty(3 ** m, dtype=np.int8)
        t_digits = np.empty(3 ** m, dtype=np.int8)
        codes = np.arange(3 ** m, dtype=np.int64)
        for j in range(m):
            d = (codes // (3 ** j)) % 3
            dmap = np.where(d == 0, -1, np.where(d == 1, 1, -1))
            if j < m:
                # first m go to bottom
                pass
        # Actually we need m bottom and m top; use 2*m slots -> 3^(2m) too big.
        # Instead: bottom from lower m trits, top from upper m trits (codes in 0..9^m-1) -> 9^m big.
        # Fallback: use product but cap.
        pass

    # Given complexity, just do a moderate direct local search from a good seed.
    # Seed: all matching +1, bottom all -1, top all -1 (usually unbalanced).
    seed = np.ones(E, dtype=np.int8)
    for i in bot_edges:
        seed[i] = -1
    for i in top_edges:
        seed[i] = -1

    consider(seed)

    # ---- Phase 2: local search (single sign flips) ----
    flip_order = np.arange(E, dtype=np.int64)
    np.random.default_rng(3).shuffle(flip_order)

    cur = seed.copy()
    cur_r = best_radius
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        for ei in flip_order:
            if time.time() > deadline - 8:
                break
            cur[ei] = -cur[ei]
            if is_balanced(cur):
                cur[ei] = -cur[ei]
                continue
            r = spectral_radius(cur)
            if r < cur_r - 1e-9:
                cur_r = r
                improved = True
                if r < best_radius:
                    best_radius = r
                    best_signs = cur.copy()
            else:
                cur[ei] = -cur[ei]

    # ---- Phase 3: multi-flip local search (flip 2 edges at a time) ----
    if time.time() < deadline - 12:
        # Generate random pairs
        rng2 = np.random.default_rng(7)
        cur2 = best_signs.copy()
        cur2_r = spectral_radius(cur2) if is_balanced(cur2) else float('inf')
        
        improved2 = True
        while improved2 and time.time() < deadline - 12:
            improved2 = False
            # Try a batch of random pairs
            for _ in range(50):
                if time.time() > deadline - 12:
                    break
                i, j = rng2.integers(0, E, size=2)
                if i == j:
                    continue
                cur2[i] = -cur2[i]
                cur2[j] = -cur2[j]
                if is_balanced(cur2):
                    cur2[i] = -cur2[i]
                    cur2[j] = -cur2[j]
                    continue
                r = spectral_radius(cur2)
                if r < cur2_r - 1e-9:
                    cur2_r = r
                    improved2 = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = cur2.copy()
                else:
                    cur2[i] = -cur2[i]
                    cur2[j] = -cur2[j]

    if best_signs is None:
        best_signs = -np.ones(E, dtype=np.int8)

    return np.asarray(best_signs, dtype=np.int8)
