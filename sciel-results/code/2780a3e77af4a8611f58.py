import time

def construct():
    start = time.time()
    deadline = start + 148  # 2s safety margin under 150s

    def paley_47():
        """Return adjacency bitmask list for the Paley graph P_47 (23-regular, C4-free)."""
        q = 47
        # Legendre symbol: a is QR mod q iff a^((q-1)/2) == 1 mod q
        adj = [0] * q
        for a in range(1, q):
            if pow(a, (q - 1) // 2, q) == 1:
                for x in range(q):
                    y = (x + a) % q
                    if y != x:
                        adj[x] |= (1 << y)
        # Symmetrize (Paley is symmetric since -1 is QR mod 47? Check: 47 ≡ 3 mod 4, so -1 is NQR.
        # Actually Paley graphs are undirected iff -1 is a QR, i.e., q ≡ 1 mod 4. 47 ≡ 3 mod 4.
        # So we need to make it undirected: edge (u,v) iff (v-u) is QR OR (u-v) is QR, i.e., iff difference is QR or NQR?
        # Standard Paley: for q ≡ 1 mod 4, -1 is QR so symmetric. For q ≡ 3 mod 4, we should use the "Paley-type" construction
        # where edge iff difference is a QR. Since -1 is NQR, if a is QR then -a is NQR, so (u,v) edge iff v-u QR means
        # (v,u) edge iff u-v QR iff -(v-u) QR which is false. So it's directed!
        # For undirected C4-free with high degree, we can take the undirected version:
        # edge iff (v-u) is QR OR (v-u) is NQR... that's all edges. Not useful.
        # Let me just use the directed Paley and symmetrize by OR both directions.
        for x in range(q):
            for y in range(q):
                if adj[x] & (1 << y):
                    adj[y] |= (1 << x)
        # Remove self-loops
        for x in range(q):
            adj[x] &= ~(1 << x)
        return adj

    def min_deg(adj, n):
        return min(adj[i].bit_count() for i in range(n))

    def verify_c4_free(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    if common & (common - 1):  # more than one bit
                        return False
        return True

    # Start with full Paley graph P_47
    full_adj = paley_47()
    n_full = 47
    full_deg = [full_adj[i].bit_count() for i in range(n_full)]
    print(f"Full P_47 min degree: {min(full_deg)}, max: {max(full_deg)}")

    best_result = None
    best_n = 0

    # Try removing k vertices to find best induced subgraph with min_deg >= n - 39
    # For n vertices, we need min_deg >= n - 39
    # Start with all 47, need min_deg >= 8. P_47 is 23-regular so this should work!
    # But wait, P_47 as symmetrized directed Paley may not be C4-free anymore.
    
    # Let me verify if the symmetrized version is C4-free
    if verify_c4_free(full_adj, n_full):
        # Great! P_47 symmetrized is C4-free
        md = min(full_deg)
        if md >= n_full - 39:  # 47-39=8
            edges = []
            for i in range(n_full):
                for j in range(i + 1, n_full):
                    if full_adj[i] & (1 << j):
                        edges.append([i, j])
            best_result = {'n': n_full, 'edges': edges}
            best_n = n_full
            print(f"Found valid graph with n=47, min_deg={md}")
    else:
        print("Symmetrized P_47 is NOT C4-free, trying induced subgraphs")
    
    # If full graph doesn't work, try removing vertices
    # Greedy: iteratively remove vertex with lowest degree to try to improve min degree of remaining
    # But removing a vertex reduces n, which changes the threshold n-39
    
    # Strategy: try all single-vertex removals, then pairs, etc.
    # For efficiency, focus on small removals
    
    if best_n < 47:
        # Try removing 1 vertex
        for rem in range(n_full):
            if time.time() > deadline:
                break
            n = n_full - 1
            if n <= best_n:
                continue
            required = n - 39
            # Build adj for induced subgraph on all vertices except rem
            verts = [v for v in range(n_full) if v != rem]
            vmap = {v: i for i, v in enumerate(verts)}
            adj = [0] * n
            for i, v in enumerate(verts):
                for j, w in enumerate(verts):
                    if i != j and full_adj[v] & (1 << w):
                        adj[i] |= (1 << j)
            degs = [adj[i].bit_count() for i in range(n)]
            if min(degs) >= required and verify_c4_free(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best_result = {'n': n, 'edges': edges}
                best_n = n
                print(f"Found valid graph with n={n} (removed vertex {rem}), min_deg={min(degs)}")
                break
    
    if best_n < 47 and time.time() < deadline - 5:
        # Try removing 2 vertices
        for r1 in range(n_full):
            for r2 in range(r1 + 1, n_full):
                if time.time() > deadline:
                    break
                n = n_full - 2
                if n <= best_n:
                    continue
                required = n - 39
                verts = [v for v in range(n_full) if v != r1 and v != r2]
                adj = [0] * n
                for i, v in enumerate(verts):
                    for j, w in enumerate(verts):
                        if i != j and full_adj[v] & (1 << w):
                            adj[i] |= (1 << j)
                degs = [adj[i].bit_count() for i in range(n)]
                if min(degs) >= required and verify_c4_free(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    best_result = {'n': n, 'edges': edges}
                    best_n = n
                    print(f"Found valid graph with n={n} (removed {r1},{r2}), min_deg={min(degs)}")
                    break
            if best_n >= n_full - 2:
                break

    # Fallback: small valid graph
    if best_result is None:
        # n=39, empty graph, min_deg=0 >= 0
        best_result = {'n': 39, 'edges': []}

    return best_result
