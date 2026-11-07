def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 55.0  # safety margin before 60s wall-clock kill

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Edge index lookup
    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(min(u, v), max(u, v))] = i

    def key(u, v):
        return (min(u, v), max(u, v))

    # Classify edges
    bot_edges = []
    top_edges = []
    match_edges = []
    for i, (u, v) in enumerate(edges):
        a, b = min(u, v), max(u, v)
        if 0 <= a < m and 0 <= b < m:
            bot_edges.append(i)
        elif a >= m and b >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)

    # Build adjacency for balance check
    adj_with_edge = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj_with_edge[u].append((v, i))
        adj_with_edge[v].append((u, i))

    def is_balanced(sgn):
        """Check if the signed graph is balanced (lift disconnected)."""
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

    # Precompute base adjacency for spectral radius
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
            return
        r = spectral_radius(sgn)
        if r < best_radius:
            best_radius = r
            best_signs = sgn.copy()

    # ---- Phase 1: Gauge-fixed enumeration ----
    # Gauge fix: set all bottom cycle signs to +1
    # Free parameters: top cycle signs (m) and matching signs (m)
    # Total: 2^(2m) possibilities, but for m=4,5,7,9 this is manageable
    # m=4: 2^8=256, m=5: 2^10=1024, m=7: 2^14=16384, m=9: 2^18=262144

    if 2 ** (2 * m) <= 500000:
        # Enumerate all combinations of top cycle and matching signs
        for code in range(2 ** (2 * m)):
            if time.time() > deadline - 5:
                break
            sgn = np.ones(E, dtype=np.int8)
            # Bottom cycle already +1
            for j in range(m):
                # Top cycle edge j
                bit = (code >> j) & 1
                sgn[top_edges[j]] = 1 if bit == 0 else -1
            for j in range(m):
                # Matching edge j
                bit = (code >> (m + j)) & 1
                sgn[match_edges[j]] = 1 if bit == 0 else -1
            consider(sgn)
    else:
        # For larger m, use random sampling with gauge fixing
        rng = np.random.default_rng(3)
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            sgn = np.ones(E, dtype=np.int8)
            for j in range(m):
                sgn[top_edges[j]] = rng.choice([-1, 1])
            for j in range(m):
                sgn[match_edges[j]] = rng.choice([-1, 1])
            consider(sgn)

    if best_signs is None:
        # Fallback
        best_signs = np.ones(E, dtype=np.int8)
        best_radius = float('inf')

    # ---- Phase 2: Local search refinement ----
    cur = best_signs.copy()
    cur_r = best_radius

    # Try flipping individual edges
    flip_order = np.arange(E, dtype=np.int64)
    np.random.default_rng(42).shuffle(flip_order)

    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for ei in flip_order:
            if time.time() > deadline - 3:
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

    # ---- Phase 3: Multi-flip search ----
    if time.time() < deadline - 3:
        rng2 = np.random.default_rng(7)
        cur2 = best_signs.copy()
        cur2_r = spectral_radius(cur2) if not is_balanced(cur2) else float('inf')

        improved2 = True
        while improved2 and time.time() < deadline - 3:
            improved2 = False
            for _ in range(100):
                if time.time() > deadline - 3:
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

    return np.asarray(best_signs, dtype=np.int8)
