"""Fresh design: exploit prism block-circulant structure.
The signed adjacency A = [[C1, M],[M^T, C2]] has eigenvalues equal to the
union of eigenvalues of B+ = (C1+C2+M)/2 and B- = (C1+C2-M)/2 (m even) or a
companion pair (m odd). Hence the spectral radius depends only on the sign
product of each cycle and the matching. Enumerate the 2^m distinct patterns,
then local-search on the 3m edge signs to fix validity (connected lift)."""
import time
import numpy as np

def _spectral_radius(graph, signs):
    n = graph['n']
    A = np.zeros((n, n))
    for (u, v), s in zip(graph['edges'], signs):
        A[u, v] = s
        A[v, u] = s
    w = np.linalg.eigvalsh(A)
    return float(max(abs(w.min()), abs(w.max())))

def _is_valid_lift(graph, signs):
    n = graph['n']
    adj = [[] for _ in range(2 * n)]
    for (u, v), s in zip(graph['edges'], signs):
        su = int(s)
        if su == 1:
            a, b = (2 * u, 2 * v)
        else:
            a, b = (2 * u + 1, 2 * v)
        adj[a].append(b)
        adj[b].append(a)
    for deg in adj:
        if len(deg) != 3:
            return False
    seen = [False] * (2 * n)
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                cnt += 1
                stack.append(y)
    return cnt == 2 * n

def _signs_for_pattern(graph, m, cycle1_prod, cycle2_prod, match_prod):
    edges = graph['edges']
    signs = np.zeros(len(edges), dtype=int)
    c1 = [i for i, (u, v) in enumerate(edges) if u < m and v < m]
    c2 = [i for i, (u, v) in enumerate(edges) if u >= m and v >= m]
    mt = [i for i, (u, v) in enumerate(edges) if (u < m and v >= m) or (u >= m and v < m)]
    # set all +1 then flip one edge to get the required product
    signs[c1] = 1
    signs[c2] = 1
    signs[mt] = 1
    if cycle1_prod == -1:
        signs[c1[0]] = -1
    if cycle2_prod == -1:
        signs[c2[0]] = -1
    if match_prod == -1:
        signs[mt[0]] = -1
    return signs

def sign_edges(graph, budget):
    t0 = time.time()
    deadline = t0 + 55.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    E = len(edges)

    c1 = [i for i, (u, v) in enumerate(edges) if u < m and v < m]
    c2 = [i for i, (u, v) in enumerate(edges) if u >= m and v >= m]
    mt = [i for i, (u, v) in enumerate(edges) if (u < m and v >= m) or (u >= m and v < m)]

    # enumerate 2^m patterns
    best_radius = float('inf')
    best_signs = None
    for mask in range(1 << m):
        if time.time() > deadline:
            break
        p1 = -1 if (mask & 1) else 1
        p2 = -1 if (mask & 2) else 1
        pm = -1 if (mask & 4) else 1
        signs = _signs_for_pattern(graph, m, p1, p2, pm)
        try:
            r = _spectral_radius(graph, signs)
        except Exception:
            continue
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    if best_signs is None:
        # fallback: all +1
        best_signs = np.ones(E, dtype=int)
        try:
            best_radius = _spectral_radius(graph, best_signs)
        except Exception:
            best_radius = float('inf')

    # local search to ensure validity while improving radius
    cur = best_signs.copy()
    cur_radius = best_radius
    if not _is_valid_lift(graph, cur):
        # try to make it valid via random flips
        rng = np.random.default_rng(3)
        for _ in range(20000):
            if time.time() > deadline:
                break
            i = int(rng.integers(E))
            cur[i] = -cur[i]
            if _is_valid_lift(graph, cur):
                try:
                    r = _spectral_radius(graph, cur)
                except Exception:
                    continue
                if r < cur_radius:
                    cur_radius = r
                    best_signs = cur.copy()
                break
        # if still invalid, do a more exhaustive fix
        if not _is_valid_lift(graph, cur):
            # brute force over the 3 "defining" edges to set products, then
            # try all 2^m assignments of the rest is too big; instead do
            # simulated-annealing style flips targeting validity
            for attempt in range(200000):
                if time.time() > deadline:
                    break
                i = int(rng.integers(E))
                cur[i] = -cur[i]
                if _is_valid_lift(graph, cur):
                    try:
                        r = _spectral_radius(graph, cur)
                    except Exception:
                        continue
                    if r < cur_radius:
                        cur_radius = r
                        best_signs = cur.copy()
                    break

    # final local search: flip one edge if it improves radius and stays valid
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(E):
            if time.time() > deadline:
                break
            cur[i] = -cur[i]
            if not _is_valid_lift(graph, cur):
                cur[i] = -cur[i]
                continue
            try:
                r = _spectral_radius(graph, cur)
            except Exception:
                cur[i] = -cur[i]
                continue
            if r < cur_radius - 1e-12:
                cur_radius = r
                best_signs = cur.copy()
                improved = True
            else:
                cur[i] = -cur[i]

    return best_signs
