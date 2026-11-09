import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    q = 5
    # Generate PG(2,5) points
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                vec = [x, y, z]
                fnz = -1
                for i in range(3):
                    if vec[i] != 0:
                        fnz = i
                        break
                if fnz == -1:
                    continue
                inv = pow(vec[fnz], q - 2, q)
                vec = [(v * inv) % q for v in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    n = len(points)
    if n != 31:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]

    # Polarity matrix
    A_mat = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int)

    # Build base adjacency (90 edges)
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = 0
            for k in range(3):
                for l in range(3):
                    val = (val + points[i][k] * A_mat[k, l] * points[j][l]) % q
            if val == 0:
                adj[i, j] = 1

    def count_c4(adj):
        A2 = adj @ adj
        # C4s = (1/4) * sum_{i<j} (A2[i,j]^2 - A2[i,j])
        # Use upper triangle
        s = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = A2[i, j]
                s += c * (c - 1)
        return s // 4

    def add_edge_cost(adj, u, v):
        # number of new C4s = number of length-3 paths u-a-b-v
        # = sum_{a in N(u)} sum_{b in N(v)} adj[a,b]
        nu = adj[u].nonzero()[0]
        nv = adj[v].nonzero()[0]
        cnt = 0
        for a in nu:
            for b in nv:
                if adj[a, b]:
                    cnt += 1
        return cnt

    # Find best extra edge
    best_edge = None
    min_cost = float('inf')
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                continue
            c = add_edge_cost(adj, u, v)
            if c < min_cost:
                min_cost = c
                best_edge = (u, v)
        if time.time() > deadline - 60:
            break

    if best_edge is None:
        # find any non-edge
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    best_edge = (u, v)
                    break
            if best_edge:
                break

    adj[best_edge[0], best_edge[1]] = 1
    adj[best_edge[1], best_edge[0]] = 1

    base_c4 = count_c4(adj)
    best_adj = adj.copy()
    best_c4 = base_c4

    # Local search: try to swap edges to reduce C4s while keeping 91 edges
    # Represent current edge set
    def get_edges(adj):
        es = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    es.append((i, j))
        return es

    current_edges = get_edges(adj)
    edge_set = set(current_edges)

    # Precompute neighbor sets for speed
    def neighbor_dict(adj):
        nd = {}
        for i in range(n):
            nd[i] = set(np.where(adj[i])[0].tolist())
        return nd

    def c4_from_edges(edge_list):
        a = np.zeros((n, n), dtype=np.int8)
        for u, v in edge_list:
            a[u, v] = 1
            a[v, u] = 1
        return count_c4(a)

    # Local search with edge swaps
    rng = random.Random(42)
    improve = True
    while improve and time.time() < deadline - 10:
        improve = False
        # try random swaps: remove edge e1, add edge e2
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            e1 = rng.choice(current_edges)
            # candidate new edges: non-edges
            # pick random non-edge
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                continue
            if (u, v) == e1:
                continue
            # apply swap
            new_edges = list(current_edges)
            new_edges.remove(e1)
            new_edges.append((u, v))
            # check validity: 91 distinct edges
            if len(set(new_edges)) != 91:
                continue
            c4 = c4_from_edges(new_edges)
            if c4 < best_c4:
                best_c4 = c4
                best_adj = np.zeros((n, n), dtype=np.int8)
                for a, b in new_edges:
                    best_adj[a, b] = 1
                    best_adj[b, a] = 1
                current_edges = new_edges
                edge_set = set(new_edges)
                improve = True
                break

    # Final polish: greedily try adding/removing to reduce C4s
    # Try each non-edge add + each edge remove that reduces C4
    if time.time() < deadline - 5:
        a = np.zeros((n, n), dtype=np.int8)
        for u, v in current_edges:
            a[u, v] = 1
            a[v, u] = 1
        cur = count_c4(a)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            es = get_edges(a)
            eset = set(es)
            for e1 in es:
                for u in range(n):
                    for v in range(u + 1, n):
                        if a[u, v]:
                            continue
                        if (u, v) == e1:
                            continue
                        # swap
                        a[e1[0], e1[1]] = 0
                        a[e1[1], e1[0]] = 0
                        a[u, v] = 1
                        a[v, u] = 1
                        c4 = count_c4(a)
                        a[u, v] = 0
                        a[v, u] = 0
                        a[e1[0], e1[1]] = 1
                        a[e1[1], e1[0]] = 1
                        if c4 < cur:
                            cur = c4
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        if cur < best_c4:
            best_c4 = cur
            best_adj = a.copy()

    # Build final edge list
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])

    if len(final_edges) != 91:
        # fallback
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]

    return final_edges
