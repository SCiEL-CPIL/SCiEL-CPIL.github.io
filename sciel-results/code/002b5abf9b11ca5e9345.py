import numpy as np
import time
from collections import deque


def _lift_connected(n, edges, signs):
    """True iff the 2-lift is connected (sign assignment is not a coboundary)."""
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        s = int(signs[idx])
        adj[u].append((v, s))
        adj[v].append((u, s))
    c = [-1] * n
    c[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        cu = c[u]
        for v, s in adj[u]:
            exp = (cu + (1 if s == -1 else 0)) % 2
            if c[v] == -1:
                c[v] = exp
                q.append(v)
            elif c[v] != exp:
                return True
    return False


def _spectral_radius_signed(n, eu, ev, signs_f):
    A = np.zeros((n, n), dtype=np.float64)
    np.add.at(A, (eu, ev), signs_f)
    np.add.at(A, (ev, eu), signs_f)
    return float(np.max(np.abs(np.linalg.eigvalsh(A))))


def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # explicit guard, well under the 60 s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    eu = np.array([e[0] for e in edges], dtype=np.int64)
    ev = np.array([e[1] for e in edges], dtype=np.int64)

    # Adjacency with edge indices
    adj = [[] for _ in range(n)]
    for idx in range(E):
        u, v = edges[idx]
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    # Spanning tree (BFS)
    in_tree = [False] * n
    tree_edge = [False] * E
    in_tree[0] = True
    q = deque([0])
    while q:
        u = q.popleft()
        for v, idx in adj[u]:
            if not in_tree[v]:
                in_tree[v] = True
                tree_edge[idx] = True
                q.append(v)

    non_tree = [i for i in range(E) if not tree_edge[i]]
    k = len(non_tree)  # cycle rank = m + 1

    best_signs = None
    best_radius = float('inf')

    # Exact search over cohomology classes.
    # A coboundary sign flip gives A' = D A D (D orthogonal), an orthogonal
    # similarity, so the spectrum -- and hence the spectral radius -- depends
    # only on the cohomology class of the signs.  Fixing tree edges to +1, each
    # class is represented bijectively by the signs on the (m+1) non-tree edges
    # (the fundamental-cycle products equal those signs).  The lift is connected
    # iff the class is non-trivial (at least one non-tree edge is -1).  Enumerating
    # these classes yields the global minimum spectral radius over all connected
    # 3-regular 2-lifts.
    total = 1 << k
    # Pre-allocate signs array and use a faster inner loop
    signs_template = np.ones(E, dtype=np.float64)
    for mask in range(1, total):  # mask == 0 is the trivial (disconnected) class
        if time.time() > deadline:
            break
        signs_f = signs_template.copy()
        for j in range(k):
            if (mask >> j) & 1:
                signs_f[non_tree[j]] = -1.0
        r = _spectral_radius_signed(n, eu, ev, signs_f)
        if r < best_radius:
            best_radius = r
            best_signs = signs_f.copy().astype(np.int64)

    # Safety fallback: random search with explicit connectivity test.
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(200000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=E)
            if not _lift_connected(n, edges, signs):
                continue
            r = _spectral_radius_signed(n, eu, ev, signs.astype(np.float64))
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    return best_signs.astype(np.int64)
