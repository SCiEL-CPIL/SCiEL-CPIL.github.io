import random
import time

# EVOLVE-BLOCK-START

def _max_tri_free(mask, adj):
    n = 18
    best = [0]
    tri = [0] * n
    for i in range(n):
        t = 0
        j = adj[i]
        while j:
            lsb = j & -j
            jj = lsb.bit_length() - 1
            j ^= lsb
            if adj[i] & adj[jj]:
                t |= lsb
        tri[i] = t

    def dfs(available, size):
        if not available:
            if size > best[0]:
                best[0] = size
            return
        if size + available.bit_count() <= best[0]:
            return
        lsb = available & -available
        v = lsb.bit_length() - 1
        dfs(available & ~lsb & ~tri[v], size + 1)
        dfs(available & ~lsb, size)

    dfs(mask, 0)
    return best[0]


def _score_adj(adj):
    n = 18
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b + 1, n):
                if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                    continue
                for d in range(c + 1, n):
                    if ((adj[a] >> d) & 1 and (adj[b] >> d) & 1 and (adj[c] >> d) & 1):
                        return (0, False)
    return (_max_tri_free((1 << 18) - 1, adj), True)


def _c5_adj(sizes):
    parts = []
    v = 0
    for s in sizes:
        parts.append(list(range(v, v + s)))
        v += s
    adj = [0] * 18
    for i in range(5):
        j = (i + 1) % 5
        for a in parts[i]:
            for b in parts[j]:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
    return adj


def construct():
    deadline = time.time() + 110.0
    rng = random.Random(12345)

    best_adj = None
    best_score = 99

    sizes = [4, 4, 4, 3, 3]
    adj = _c5_adj(sizes)
    s, ok = _score_adj(adj)
    if ok:
        best_adj = adj
        best_score = s

    for _ in range(500):
        if time.time() > deadline:
            break
        i, j = rng.sample(range(5), 2)
        if sizes[i] > 1:
            sizes[i] -= 1
            sizes[j] += 1
            adj = _c5_adj(sizes)
            s, ok = _score_adj(adj)
            if ok and s < best_score:
                best_score = s
                best_adj = adj
            else:
                sizes[i] += 1
                sizes[j] -= 1

    edges = []
    for i in range(18):
        for j in range(i + 1, 18):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

# EVOLVE-BLOCK-END
