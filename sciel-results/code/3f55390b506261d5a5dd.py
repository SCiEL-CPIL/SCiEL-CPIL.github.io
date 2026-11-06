"""SA-based search for 5-colourings of K_n with no monochromatic C4."""
import time
import random

_DEADLINE = 150.0


def _build_color_adjacency(colors, n, k=5):
    adj = [[0] * n for _ in range(k)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    return adj


def _count_c4(adj, n, k=5):
    total = 0
    for c in range(k):
        ac = adj[c]
        for u in range(n):
            mu = ac[u]
            for v in range(u + 1, n):
                b = (mu & ac[v]).bit_count()
                if b >= 2:
                    total += b * (b - 1) // 2
    return total


def _sa(n, k=5, time_budget=20.0, seed=0):
    rng = random.Random(seed)
    m = n * (n - 1) // 2
    t_end = time.time() + time_budget
    edge_uv = [(i, j) for i in range(n) for j in range(i + 1, n)]

    colors = [rng.randrange(k) for _ in range(m)]
    adj = _build_color_adjacency(colors, n, k)
    cost = _count_c4(adj, n, k)
    best_colors = colors[:]
    best_cost = cost

    T = 8.0
    T_min = 0.003
    cool = 0.99998
    while time.time() < t_end:
        e = rng.randrange(m)
        u, v = edge_uv[e]
        c_old = colors[e]
        c_new = rng.randrange(k)
        if c_new == c_old:
            continue
        colors[e] = c_new
        adj[c_old][u] &= ~(1 << v)
        adj[c_old][v] &= ~(1 << u)
        adj[c_new][u] |= (1 << v)
        adj[c_new][v] |= (1 << u)
        new_cost = _count_c4(adj, n, k)
        delta = new_cost - cost
        if delta <= 0 or rng.random() < 2.718281828 ** (-delta / max(T, 1e-9)):
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors
        else:
            colors[e] = c_old
            adj[c_new][u] &= ~(1 << v)
            adj[c_new][v] &= ~(1 << u)
            adj[c_old][u] |= (1 << v)
            adj[c_old][v] |= (1 << u)
        T = max(T * cool, T_min)

    if best_cost == 0:
        return best_colors
    return None


def _verify(colors, n, k=5):
    adj = _build_color_adjacency(colors, n, k)
    return _count_c4(adj, n, k) == 0


def construct():
    start = time.time()
    deadline = start + _DEADLINE - 5.0

    best_n = 0
    best_colors = None

    for n in range(26, 10, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        budget = min(25.0, remaining - 3.0)
        if budget < 2.0:
            break
        found = None
        for seed in (0, 1, 2, 3):
            if time.time() > deadline - 2.0:
                break
            b = min(deadline - 2.0 - time.time(), budget)
            if b < 1.0:
                break
            found = _sa(n, k=5, time_budget=b, seed=seed)
            if found is not None:
                break
        if found is not None:
            if _verify(found, n):
                best_n = n
                best_colors = found
                break

    if best_n == 0 or best_colors is None:
        n = 11
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
