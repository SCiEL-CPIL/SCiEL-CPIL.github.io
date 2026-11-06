import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def check_c4_safe(adj, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks."""
        mask_u = adj[u]
        mask_v = adj[v]
        m = mask_u
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            m ^= lsb
            if adj[a] & mask_v:
                return False
        return True

    def verify_graph(adj, n, min_deg_req):
        """Verify C4-free and min degree."""
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg_req:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    if common & (common - 1):
                        return False
        return True

    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    def local_improve(adj, deg, n, min_deg_req, max_iters=500):
        """Try to fix min-degree violations via edge swaps."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Check if all degrees meet requirement
            min_d = min(deg[i] for i in range(n))
            if min_d >= min_deg_req:
                return True, adj, deg

            # Find vertex with minimum degree
            u = min(range(n), key=lambda i: deg[i])

            # Try to add an edge from u to a non-neighbor
            added = False
            for v in range(n):
                if v == u or (adj[u] >> v) & 1:
                    continue
                if check_c4_safe(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    added = True
                    break

            if not added:
                # Try edge swap: remove (u,a), add (u,v)
                m = adj[u]
                neighbors = []
                while m:
                    lsb = m & (-m)
                    a = lsb.bit_length() - 1
                    m ^= lsb
                    neighbors.append(a)

                for a in neighbors:
                    for v in range(n):
                        if v == u or v == a or (adj[u] >> v) & 1:
                            continue
                        # Check if removing (u,a) and adding (u,v) keeps C4-free
                        # Temporarily remove (u,a)
                        adj[u] &= ~(1 << a)
                        adj[a] &= ~(1 << u)
                        if check_c4_safe(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            deg[u] = deg[u] - 1 + 1  # net zero
                            deg[a] -= 1
                            deg[v] += 1
                            added = True
                            break
                        else:
                            # Restore
                            adj[u] |= (1 << a)
                            adj[a] |= (1 << u)
                    if added:
                        break

            if not added:
                break

        min_d = min(deg[i] for i in range(n))
        return min_d >= min_deg_req, adj, deg

    # Try from high n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break

        min_deg_req = n - 39
        if min_deg_req <= 0:
            return {'n': n, 'edges': []}

        found = False
        best_edges = []

        # Multiple strategies with different random seeds
        for strategy in range(80):
            if time.time() > deadline:
                break

            random.seed(strategy * 7919 + n * 31)

            adj = [0] * n
            deg = [0] * n

            # Phase 1: Greedy edge addition
            stuck = False
            max_steps = 300
            steps = 0

            while steps < max_steps:
                low_verts = [i for i in range(n) if deg[i] < min_deg_req]
                if not low_verts:
                    break

                # Pick the vertex with the lowest degree among low_verts
                u = min(low_verts, key=lambda i: deg[i])

                # Find safe neighbors
                safe_vs = []
                for v in range(n):
                    if v == u or (adj[u] >> v) & 1:
                        continue
                    if check_c4_safe(adj, u, v):
                        safe_vs.append(v)

                if not safe_vs:
                    stuck = True
                    break

                # Pick v: prefer lowest degree to balance
                v = min(safe_vs, key=lambda x: deg[x])

                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                steps += 1

            # Phase 2: Local improvement
            if not stuck:
                success, adj, deg = local_improve(adj, deg, n, min_deg_req, max_iters=200)
                if success and verify_graph(adj, n, min_deg_req):
                    found = True
                    best_edges = extract_edges(adj, n)
                    break
            else:
                # Even if stuck, try local improvement
                success, adj, deg = local_improve(adj, deg, n, min_deg_req, max_iters=300)
                if success and verify_graph(adj, n, min_deg_req):
                    found = True
                    best_edges = extract_edges(adj, n)
                    break

        if found:
            return {'n': n, 'edges': best_edges}

    # Fallback
    return {'n': 39, 'edges': []}
