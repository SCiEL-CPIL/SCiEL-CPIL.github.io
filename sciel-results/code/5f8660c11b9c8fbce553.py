import time
import random
import numpy as np

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 6.0 - (time.time() - _START)


def _edge_index(n):
    idx = {}
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[(i, j)] = k
            k += 1
    return idx


def _find_c6(adj):
    n = len(adj)
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if not adj[c][d] or not adj[b][d]:
                        continue
                    for e in range(d + 1, n):
                        if not adj[d][e] or not adj[c][e]:
                            continue
                        for f in range(e + 1, n):
                            if adj[e][f] and adj[a][f]:
                                return (a, b, c, d, e, f)
    return None


def _check_coloring(n, colors):
    adj = [None] * 4
    for c in range(4):
        adj[c] = np.zeros((n, n), dtype=bool)
    for (i, j), col in enumerate(colors):
        adj[col][i][j] = True
        adj[col][j][i] = True
    for c in range(4):
        if _find_c6(adj[c].tolist()) is not None:
            return False
    return True


def _spectral_embed(n, seed):
    rng = np.random.default_rng(seed)
    k = 2 * (n - 1)
    while True:
        G = np.zeros((n, n))
        for i in range(n):
            perm = rng.permutation(n)
            for j in perm[:k]:
                if i != j:
                    G[i][j] = 1.0
        if np.allclose(np.sum(G, axis=1), k):
            break
    L = np.diag(G.sum(axis=1)) - G
    w, V = np.linalg.eigh(L)
    order = np.argsort(w)
    order = order[1:]  # skip trivial
    X = V[:, order[:3]]
    return X


def _color_from_embed(X, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    A = X.T @ X
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = A[i, j]
    # project to 4-tetrahedron vertices by nearest
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ])
    # use pairwise similarity to assign: color = argmax over vertices of sum of similarities
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # similarity in [0,1]
            sim = 0.5 * (1 + A[i, j])
            # pick color whose "preferred distance" matches; map sim to one of 4
            # use 4 quantiles of Q to balance colors
            colors.append(0)
    # simpler: assign each edge a color by hashing a rotation of the sim value
    # We instead do a balanced assignment via sorting similarities
    flat = []
    for i in range(n):
        for j in range(i + 1, n):
            flat.append((0.5 * (1 + A[i, j]), i, j))
    flat.sort(key=lambda t: t[0])
    m = len(flat)
    colors = [0] * m
    # assign colors in round-robin over sorted order to balance, but keep low-sim together
    for t in range(m):
        colors[t] = t % 4
    # map sorted position back to lexicographic order
    col_by_pos = {}
    for t, (s, i, j) in enumerate(flat):
        col_by_pos[(i, j)] = colors[t]
    out = [col_by_pos[(i, j)] for i in range(n) for j in range(i + 1, n)]
    return out


def _local_search(n, colors, iters, seed):
    idx = _edge_index(n)
    rng = random.Random(seed)
    best = list(colors)

    def adj_list(col):
        a = np.zeros((n, n), dtype=bool)
        for (i, j), c in enumerate(best):
            if c == col:
                a[i][j] = True
                a[j][i] = True
        return a

    for it in range(iters):
        if _remaining() < 10:
            break
        # find a C6 in some color
        target = None
        for c in range(4):
            cyc = _find_c6(adj_list(c).tolist())
            if cyc is not None:
                target = (c, cyc)
                break
        if target is None:
            return best
        c, cyc = target
        # pick a random edge on the cycle to recolor
        e = rng.randrange(6)
        i, j = cyc[e], cyc[(e + 1) % 6]
        if i > j:
            i, j = j, i
        # choose new color that does not immediately create a C6 (try all, keep best)
        cur = best[idx[(i, j)]]
        for nc in range(4):
            if nc == cur:
                continue
            trial = list(best)
            trial[idx[(i, j)]] = nc
            if _check_coloring(n, trial):
                best = trial
                break
        else:
            # random perturbation to escape
            p = rng.randrange(len(best))
            best[p] = rng.randrange(4)
    return best


def _solve(n, seed):
    if _remaining() < 15:
        return None
    X = _spectral_embed(n, seed)
    colors = _color_from_embed(X, seed)
    iters = 4000
    res = _local_search(n, colors, iters, seed)
    if res is not None and _check_coloring(n, res):
        return res
    return None


def construct():
    # known-good fallback (n=9)
    def fallback():
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    best = fallback()
    for n in range(17, 9, -1):
        if _remaining() < 20:
            break
        found = None
        for s in range(60):
            if _remaining() < 15:
                break
            r = _solve(n, 1000 + s)
            if r is not None:
                found = r
                break
        if found is not None:
            return {'n': n, 'colors': found}
    return best
