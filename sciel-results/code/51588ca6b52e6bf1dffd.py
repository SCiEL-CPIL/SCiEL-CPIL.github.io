import random
import time

def _check_mask(adj, mask):
    """Return True if the subset `mask` induces a regular graph (all equal degrees)."""
    k = mask.bit_count()
    if k < 6:
        return False
    first = mask & (-mask)
    ref = (adj[first] & mask).bit_count()
    m = mask >> 1
    while m:
        lsb = m & (-m)
        if (adj[lsb] & mask).bit_count() != ref:
            return False
        m ^= lsb
    return True

def _has_regular_subset(adj, n):
    """Return True if any subset of size >= 6 induces a regular graph."""
    for size in range(6, n + 1):
        comb = []
        for i in range(n - size, -1, -1):
            comb.append((1 << i) | (comb[i + 1] & ((1 << (i + 1)) - 1)))
        for mask in comb:
            if _check_mask(adj, mask):
                return True
    return False

def _is_valid(adj, n):
    return not _has_regular_subset(adj, n)

def _random_adj(n, p, rng):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj

def _flip_edges(adj, n, k, rng):
    for _ in range(k):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i != j and (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= 1 << j
            adj[j] |= 1 << i

def _to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    deadline = time.time() + 110.0
    rng = random.Random(12345)

    best = {'n': 5, 'edges': []}

    for n in range(22, 4, -1):
        if time.time() > deadline - 20.0:
            break
        found = False
        for attempt in range(400000):
            if time.time() > deadline - 10.0:
                break
            adj = _random_adj(n, 0.5, rng)
            if _is_valid(adj, n):
                best = {'n': n, 'edges': _to_edges(adj, n)}
                found = True
                break
        if found:
            break

    # If no fully-valid graph found, fall back to the largest n where we found one,
    # or keep a smaller valid one. Ensure we return a valid graph.
    return best
