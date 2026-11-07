def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 70.0  # safety margin before 90s wall-clock kill

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
    top_edges = []
    top_edge_map = {}  # maps top vertex to its edge index in the cycle
    match_edges = []
    for i, (u, v) in enumerate(edges):
        a, b = min(u, v), max(u, v)
        if 0 <= a < m and 0 <= b < m:
            bot_edges.append(i)
        elif a >= m and b >= m:
            top_edges.append(i)
            # For top cycle, store which vertices this edge connects (both >= m)
        else:
            match_edges.append(i)

    # Build adjacency (for balance check)
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Precompute edge-to-vertex mapping for faster balance check
    adj_with_edge = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj_with_edge[u].append((v, i))
        adj_with_edge[v].append((u, i))

    def is_balanced(sgn):
        """Check if the lift is connected (no balanced cycles that would disconnect it).
        A lift is disconnected iff there exists a non-trivial potential function pot: V -> {+1,-1}
        such that pot[u]*pot[v] = sgn[uv] for all edges. This is equivalent to the signed graph
        being 'balanced' in the sense that all cycles have product +1... wait, no.
        
        Actually, for a 2-lift to be connected, the signed graph must NOT be balanced.
        A signed graph is balanced if and only if its 2-lift is disconnected (has two components).
        So we need the lift to be connected, which means the signed graph must be UNBALANCED.
        
        The standard test: assign potentials. Start with pot[0] = +1. For each edge (u,v) with sign s,
        we require pot[v] = pot[u] * s. If we can assign consistent potentials to all vertices,
        the graph is balanced and the lift is disconnected. If we get a contradiction, the graph
        is unbalanced and the lift is connected.
        """
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
                    # Contradiction found -> unbalanced -> connected lift
                    return True
        # No contradiction -> balanced -> disconnected lift
        return False

    # Precompute base adjacency matrix structure
    # For spectral radius computation, we build the signed adjacency matrix each time
    # but we can optimize by precomputing the structure

    def spectral_radius(sgn):
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(sgn[i])
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    best_signs = None
    best_radius = float('inf')

    def consider(sgn):
        nonlocal best_signs, best_radius
        if not is_balanced(sgn):
            return  # disconnected lift -> invalid
        r = spectral_radius(sgn)
        if r < best_radius:
            best_radius = r
            best_signs = sgn.copy()

    # ---- Strategy: Exploit the prism structure ----
    # The prism graph has two m-cycles connected by a perfect matching.
    # Key insight: The matching edges can be used as a gauge. 
    # If we fix all matching signs to +1, we can still achieve any lift connectivity
    # by adjusting the cycle signs appropriately.
    # 
    # Actually, let's think more carefully. The signed adjacency matrix of the base graph
    # has the form:
    #   [B  M]
    #   [M  T]
    # where B is the bottom cycle signed adjacency, T is the top cycle signed adjacency,
    # and M is the matching signed adjacency (diagonal if we order vertices as 0..m-1, m..2m-1).
    #
    # For the lift to be connected, the signed graph must be unbalanced.
    #
    # Let's try a systematic approach:
    # 1. Start with matching all +1.
    # 2. The bottom cycle signs and top cycle signs determine the spectrum.
    # 3. We need at least one "frustrated" cycle to ensure connectivity.
    
    # Seed: all +1 except make one cycle frustrated
    # A frustrated cycle has product of signs = -1
    
    seed = np.ones(E, dtype=np.int8)
    # Make bottom cycle frustrated: flip one edge
    if len(bot_edges) > 0:
        seed[bot_edges[0]] = -1
    consider(seed)

    # Try various seeds
    # Seed 2: top cycle frustrated
    seed2 = np.ones(E, dtype=np.int8)
    if len(top_edges) > 0:
        seed2[top_edges[0]] = -1
    consider(seed2)

    # Seed 3: both cycles frustrated
    seed3 = np.ones(E, dtype=np.int8)
    if len(bot_edges) > 0:
        seed3[bot_edges[0]] = -1
    if len(top_edges) > 0:
        seed3[top_edges[0]] = -1
    consider(seed3)

    # Seed 4: matching frustrated (flip one matching edge)
    seed4 = np.ones(E, dtype=np.int8)
    if len(match_edges) > 0:
        seed4[match_edges[0]] = -1
    consider(seed4)

    # ---- Local search: single sign flips ----
    flip_order = np.arange(E, dtype=np.int64)
    rng = np.random.default_rng(3)
    rng.shuffle(flip_order)

    cur = best_signs.copy()
    cur_r = best_radius
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for ei in flip_order:
            if time.time() > deadline - 15:
                break
            cur[ei] = -cur[ei]
            if is_balanced(cur):
                r = spectral_radius(cur)
                if r < cur_r - 1e-9:
                    cur_r = r
                    improved = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = cur.copy()
                else:
                    cur[ei] = -cur[ei]
            else:
                cur[ei] = -cur[ei]

    # ---- Multi-flip local search: flip 2 edges at a time ----
    if time.time() < deadline - 20:
        rng2 = np.random.default_rng(7)
        cur2 = best_signs.copy()
        cur2_r = spectral_radius(cur2) if is_balanced(cur2) else float('inf')
        
        improved2 = True
        attempts = 0
        while improved2 and time.time() < deadline - 20 and attempts < 5000:
            improved2 = False
            attempts += 1
            # Try random pairs
            for _ in range(30):
                if time.time() > deadline - 20:
                    break
                i, j = rng2.integers(0, E, size=2)
                if i == j:
                    continue
                cur2[i] = -cur2[i]
                cur2[j] = -cur2[j]
                if is_balanced(cur2):
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
                else:
                    cur2[i] = -cur2[i]
                    cur2[j] = -cur2[j]

    # ---- Final polish: try to reduce radius with small perturbations ----
    if time.time() < deadline - 5 and best_signs is not None:
        rng3 = np.random.default_rng(42)
        cur3 = best_signs.copy()
        cur3_r = best_radius
        
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            # Flip 1 or 2 random edges
            k = rng3.integers(1, 3)
            flips = rng3.integers(0, E, size=k)
            unique_flips = list(set(flips.tolist()))
            for f in unique_flips:
                cur3[f] = -cur3[f]
            if is_balanced(cur3):
                r = spectral_radius(cur3)
                if r < cur3_r - 1e-9:
                    cur3_r = r
                    if r < best_radius:
                        best_radius = r
                        best_signs = cur3.copy()
                else:
                    for f in unique_flips:
                        cur3[f] = -cur3[f]
            else:
                for f in unique_flips:
                    cur3[f] = -cur3[f]

    if best_signs is None:
        # Fallback: all +1 except one edge to ensure connectivity
        best_signs = np.ones(E, dtype=np.int8)
        if E > 0:
            best_signs[0] = -1

    return np.asarray(best_signs, dtype=np.int8)
