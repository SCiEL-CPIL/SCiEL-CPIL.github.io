"""Program for spectral graph lifts on prism graphs."""
import numpy as np


def _build_signed_adj(graph, signs):
    """Build the signed adjacency matrix of the base graph from signs.

    graph: {'n': 2m, 'edges': [(u,v), ...]}
    signs: array of shape (n_edges,) with values in {-1,+1} (or continuous).
    Returns the signed adjacency matrix A (n x n), symmetric.
    """
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A


def _spectral_radius(A):
    """Max absolute eigenvalue of symmetric A."""
    vals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(vals)))


def _is_connected_lift(graph, signs):
    """Check whether the 2-lift is connected.

    The 2-lift is connected iff there exists at least one cycle in the base
    graph whose product of edge signs is -1. For a prism graph the relevant
    fundamental cycles are: the top cycle, the bottom cycle, and the
    'rung' cycles. We check via a parity/labeling argument: the lift splits
    into two components iff the signs are 'consistent' (product +1 around
    every cycle). We detect this by attempting to assign each vertex a side
    label in {0,1} such that for every edge (u,v) with sign s, the labels
    satisfy label[v] == label[u] if s>0 else 1-label[u]. If such a labeling
    exists, the lift is disconnected (2 components). Otherwise connected.
    """
    n = graph['n']
    edges = graph['edges']
    # Build adjacency with signs
    adj = {i: [] for i in range(n)}
    for (u, v), s in zip(edges, signs):
        adj[u].append((v, s))
        adj[v].append((u, s))

    label = [-1] * n
    # BFS from vertex 0
    from collections import deque
    label[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for (v, s) in adj[u]:
            expected = label[u] if s > 0 else 1 - label[u]
            if label[v] == -1:
                label[v] = expected
                q.append(v)
            else:
                if label[v] != expected:
                    # Inconsistent -> no global labeling -> lift is connected
                    return True
    # Consistent labeling found -> lift is disconnected
    return False


def sign_edges(graph, budget):
    """Return one sign (-1 or +1) per edge, minimizing spectral radius
    of the signed adjacency matrix, subject to the 2-lift being connected."""
    rng = np.random.default_rng(3)
    n_edges = len(graph['edges'])
    m = graph['n'] // 2

    def eval_signs(signs):
        A = _build_signed_adj(graph, signs)
        return _spectral_radius(A)

    # ---- Continuous relaxation: optimize over [-1,1] via gradient descent ----
    # Parameterize signs as continuous values; minimize spectral radius +
    # a small penalty to push values toward +/-1 (bilinear/soft sign).
    def objective(x):
        A = _build_signed_adj(graph, x)
        vals = np.linalg.eigvalsh(A)
        lam_max = np.max(np.abs(vals))
        # Encourage |x| -> 1 (soft sign) to make rounding meaningful
        penalty = 1e-3 * np.sum((1.0 - x * x) ** 2)
        return lam_max + penalty

    # Gradient via finite differences (small problem, cheap enough)
    def gradient(x):
        g = np.zeros_like(x)
        h = 1e-4
        base = objective(x)
        for i in range(n_edges):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (objective(xp) - objective(xm)) / (2 * h)
        return g

    # Start from a few random continuous initializations
    best_x = None
    best_obj = np.inf
    for trial in range(8):
        x = rng.uniform(-1.0, 1.0, size=n_edges)
        # Gradient descent
        lr = 0.05
        for step in range(300):
            g = gradient(x)
            x = x - lr * g
            # Clip to [-1,1]
            x = np.clip(x, -1.0, 1.0)
            if step % 20 == 0:
                obj = objective(x)
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.copy()
        obj = objective(x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()

    # ---- Round continuous solution to +/-1 ----
    signs = np.where(best_x >= 0, 1.0, -1.0).astype(int)

    # ---- Local search: flip individual edges to reduce spectral radius,
    #      while maintaining connectivity ----
    def valid(signs):
        return _is_connected_lift(graph, signs)

    if not valid(signs):
        # Find a flip that makes it connected; otherwise random search
        improved = False
        for i in range(n_edges):
            cand = signs.copy()
            cand[i] = -cand[i]
            if valid(cand):
                signs = cand
                improved = True
                break
        if not improved:
            # Random restart until connected
            for _ in range(2000):
                cand = rng.choice([-1, 1], size=n_edges).astype(int)
                if valid(cand):
                    signs = cand
                    break

    # Local search loop
    current_val = eval_signs(signs)
    improved = True
    max_iters = 20000
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        # Try flipping each edge
        for i in range(n_edges):
            cand = signs.copy()
            cand[i] = -cand[i]
            if not valid(cand):
                continue
            v = eval_signs(cand)
            if v < current_val - 1e-12:
                signs = cand
                current_val = v
                improved = True
                break  # restart scan after improvement

    # Also try a few random restarts + local search to escape local minima
    best_signs = signs.copy()
    best_val = current_val
    for restart in range(30):
        cand = rng.choice([-1, 1], size=n_edges).astype(int)
        if not valid(cand):
            # force connectivity with a single flip
            for i in range(n_edges):
                c2 = cand.copy(); c2[i] = -c2[i]
                if valid(c2):
                    cand = c2
                    break
            else:
                continue
        val = eval_signs(cand)
        # local search from this start
        for _ in range(2000):
            imp = False
            for i in range(n_edges):
                c2 = cand.copy()
                c2[i] = -c2[i]
                if not valid(c2):
                    continue
                v2 = eval_signs(c2)
                if v2 < val - 1e-12:
                    cand = c2
                    val = v2
                    imp = True
                    break
            if not imp:
                break
        if val < best_val:
            best_val = val
            best_signs = cand.copy()

    return best_signs.astype(int)
