"""Local search over edge sign flips to minimize the lift's spectral radius."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    t0 = time.time()
    DEADLINE = 82.0  # safety margin under the 90 s wall-clock kill

    m = graph['n'] // 2
    edges = graph['edges']
    E = len(edges)
    n = graph['n']

    # Build the (symmetric) adjacency matrix of the base graph.
    A = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0

    def spectral_radius(signs):
        # signed adjacency matrix, symmetric
        S = A * 0.0
        for e, (u, v) in enumerate(edges):
            s = signs[e]
            S[u, v] = s
            S[v, u] = s
        # spectral radius = max |eigenvalue|
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def is_connected(signs):
        # 2-lift: vertices (v,0),(v,1). +1 joins equal sides, -1 joins opposite.
        nn = 2 * n
        adj = [[] for _ in range(nn)]
        for e, (u, v) in enumerate(edges):
            s = signs[e]
            if s > 0:
                a, b = (2 * u, 2 * v)
            else:
                a, b = (2 * u, 2 * v + 1)
            adj[a].append(b)
            adj[b].append(a)
        # BFS
        seen = [False] * nn
        seen[0] = True
        stack = [0]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        return all(seen)

    def objective(signs):
        if not is_connected(signs):
            return float('inf')
        return spectral_radius(signs)

    # ---- initial sign vector (all +1), then a few random restarts ----
    rng = np.random.default_rng(3)

    def local_search(signs):
        signs = signs.copy()
        cur = objective(signs)
        if cur == float('inf'):
            return signs, cur
        improved = True
        while improved:
            improved = False
            # randomize the order of edges to break ties
            order = rng.permutation(E)
            for e in order:
                if time.time() > DEADLINE:
                    return signs, cur
                signs[e] = -signs[e]
                val = objective(signs)
                if val < cur - 1e-12:
                    cur = val
                    improved = True
                else:
                    signs[e] = -signs[e]
        return signs, cur

    best_signs = None
    best_val = float('inf')

    # Start from all +1
    s0 = np.ones(E, dtype=int)
    s0, v0 = local_search(s0)
    if v0 < best_val:
        best_val = v0
        best_signs = s0.copy()

    # Random restarts
    restarts = 0
    while time.time() < DEADLINE:
        restarts += 1
        s = rng.choice([-1, 1], size=E)
        s, v = local_search(s)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
        # early stop if we hit a very good value
        if best_val < 1.0:
            break

    # Final high-precision polish: repeated full sweeps with tighter tolerance
    if best_signs is not None and time.time() < DEADLINE:
        s = best_signs.copy()
        cur = objective(s)
        for _ in range(5):
            if time.time() > DEADLINE:
                break
            improved = True
            while improved:
                improved = False
                for e in range(E):
                    if time.time() > DEADLINE:
                        break
                    s[e] = -s[e]
                    val = objective(s)
                    if val < cur - 1e-10:
                        cur = val
                        improved = True
                    else:
                        s[e] = -s[e]
        if cur < best_val:
            best_val = cur
            best_signs = s

    if best_signs is None:
        best_signs = np.ones(E, dtype=int)

    return best_signs
# EVOLVE-BLOCK-END
