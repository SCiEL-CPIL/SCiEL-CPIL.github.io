def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 55.0

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(min(u, v), max(u, v))] = i

    def key(u, v):
        return (min(u, v), max(u, v))

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

    A_base = np.zeros((n, n))
    for (u, v) in edges:
        A_base[u, v] = 1
        A_base[v, u] = 1

    edge_uv = np.array([(u, v) for (u, v) in edges], dtype=np.int32)

    def build_A(sgn):
        A = A_base.copy()
        for i, (u, v) in enumerate(edges):
            s = sgn[i]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(sgn):
        A = build_A(sgn)
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    def spectral_radius_with_eigvec(sgn):
        A = build_A(sgn)
        w, V = np.linalg.eigh(A)
        max_idx = np.argmax(np.abs(w))
        return float(abs(w[max_idx])), V[:, max_idx]

    def edge_gradients(sgn, lam, v):
        A = build_A(sgn)
        sign_lam = 1.0 if lam >= 0 else -1.0
        grad_A = sign_lam * np.outer(v, v)
        grad_edge = np.zeros(E)
        for i, (u, v) in enumerate(edges):
            grad_edge[i] = grad_A[u, v]
        return grad_edge

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

    # Phase 1: Gauge-fixed systematic search
    # Fix bottom edges to +1, vary top and matching
    num_top = len(top_edges)
    num_match = len(match_edges)
    total_combos = 2 ** (num_top + num_match)

    if total_combos <= 200000:
        for code in range(total_combos):
            if time.time() > deadline - 8:
                break
            sgn = np.ones(E, dtype=np.int8)
            for j in range(num_top):
                bit = (code >> j) & 1
                sgn[top_edges[j]] = 1 if bit == 0 else -1
            for j in range(num_match):
                bit = (code >> (num_top + j)) & 1
                sgn[match_edges[j]] = 1 if bit == 0 else -1
            consider(sgn)
    else:
        rng = np.random.default_rng(3)
        # Smart random sampling with bias toward low-radius configurations
        for _ in range(50000):
            if time.time() > deadline - 8:
                break
            sgn = np.ones(E, dtype=np.int8)
            for j in range(num_top):
                sgn[top_edges[j]] = rng.choice([-1, 1])
            for j in range(num_match):
                sgn[match_edges[j]] = rng.choice([-1, 1])
            consider(sgn)

    if best_signs is None:
        best_signs = np.ones(E, dtype=np.int8)
        best_radius = spectral_radius(best_signs)

    # Phase 2: Multi-start local search with gradient-guided flip order
    def local_search(sgn0, max_iters=2000):
        cur = sgn0.copy()
        if is_balanced(cur):
            return None, None
        cur_r = spectral_radius(cur)
        if cur_r >= best_radius:
            return None, None

        # Get gradient for ordering
        _, eigvec = spectral_radius_with_eigvec(cur)
        A = build_A(cur)
        w = np.linalg.eigvalsh(A)
        max_idx = np.argmax(np.abs(w))
        lam = w[max_idx]
        grads = edge_gradients(cur, lam, eigvec)

        flip_order = np.argsort(np.abs(grads))[::-1]  # largest gradient first

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            for ei in flip_order:
                if time.time() > deadline - 5:
                    return cur, cur_r
                cur[ei] = -cur[ei]
                if is_balanced(cur):
                    cur[ei] = -cur[ei]
                    continue
                r = spectral_radius(cur)
                if r < cur_r - 1e-10:
                    cur_r = r
                    improved = True
                    # Update gradient periodically
                    if iters % 10 == 0:
                        _, eigvec2 = spectral_radius_with_eigvec(cur)
                        w2 = np.linalg.eigvalsh(build_A(cur))
                        mi2 = np.argmax(np.abs(w2))
                        lam2 = w2[mi2]
                        grads = edge_gradients(cur, lam2, eigvec2)
                        flip_order = np.argsort(np.abs(grads))[::-1]
                else:
                    cur[ei] = -cur[ei]
        return cur, cur_r

    # Multiple restarts from different starting points
    if time.time() < deadline - 15:
        rng_ms = np.random.default_rng(123)
        starts = [best_signs.copy()]
        for _ in range(20):
            s = np.ones(E, dtype=np.int8)
            for j in range(num_top):
                s[top_edges[j]] = rng_ms.choice([-1, 1])
            for j in range(num_match):
                s[match_edges[j]] = rng_ms.choice([-1, 1])
            starts.append(s)

        for s0 in starts:
            if time.time() > deadline - 15:
                break
            res_sgn, res_r = local_search(s0)
            if res_sgn is not None and res_r < best_radius:
                best_radius = res_r
                best_signs = res_sgn.copy()

    # Phase 3: 2-opt local search
    if time.time() < deadline - 10:
        cur = best_signs.copy()
        cur_r = best_radius
        rng2 = np.random.default_rng(77)
        improved2 = True
        while improved2 and time.time() < deadline - 10:
            improved2 = False
            for _ in range(500):
                if time.time() > deadline - 10:
                    break
                i, j = rng2.integers(0, E, size=2)
                if i == j:
                    continue
                cur[i] = -cur[i]
                cur[j] = -cur[j]
                if is_balanced(cur):
                    cur[i] = -cur[i]
                    cur[j] = -cur[j]
                    continue
                r = spectral_radius(cur)
                if r < cur_r - 1e-10:
                    cur_r = r
                    improved2 = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = cur.copy()
                else:
                    cur[i] = -cur[i]
                    cur[j] = -cur[j]

    # Phase 4: Simulated annealing-style escape
    if time.time() < deadline - 8:
        cur = best_signs.copy()
        cur_r = best_radius
        rng3 = np.random.default_rng(555)
        T = 0.1
        for sa_iter in range(2000):
            if time.time() > deadline - 8:
                break
            ei = rng3.integers(0, E)
            cur[ei] = -cur[ei]
            if is_balanced(cur):
                cur[ei] = -cur[ei]
                continue
            r = spectral_radius(cur)
            delta_r = r - cur_r
            if delta_r < 0 or rng3.random() < np.exp(-delta_r / max(T, 1e-6)):
                cur_r = r
                if r < best_radius:
                    best_radius = r
                    best_signs = cur.copy()
            else:
                cur[ei] = -cur[ei]
            T *= 0.995

    # Phase 5: Final precise polish
    if time.time() < deadline - 2:
        cur5 = best_signs.copy()
        cur5_r = best_radius
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            best_flip = None
            best_new_r = cur5_r
            for ei in range(E):
                if time.time() > deadline - 2:
                    break
                cur5[ei] = -cur5[ei]
                if is_balanced(cur5):
                    cur5[ei] = -cur5[ei]
                    continue
                r = spectral_radius(cur5)
                if r < best_new_r - 1e-12:
                    best_new_r = r
                    best_flip = ei
                cur5[ei] = -cur5[ei]
            if best_flip is not None:
                cur5[best_flip] = -cur5[best_flip]
                cur5_r = best_new_r
                if cur5_r < best_radius:
                    best_radius = cur5_r
                    best_signs = cur5.copy()
            else:
                break

    return np.asarray(best_signs, dtype=np.int8)
