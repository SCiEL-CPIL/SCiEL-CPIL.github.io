"""Covering code K_3(6,1): coset partition + exact minimum vertex cover on quotient graph."""
# EVOLVE-BLOCK-START
import time
import random
import collections
from itertools import product


def _to_tuple(x, q, n):
    out = []
    for _ in range(n):
        out.append(x % q)
        x //= q
    return out


def _gen_code(q, n):
    """Random [n,3,d] code; prefer min distance >= 4."""
    last = None
    for _ in range(8000):
        gens = [list(random.randint(0, q - 1) for _ in range(n)) for _ in range(3)]
        code = [tuple(sum((g[i] * c[i] for i in range(n)) % q for c in zip(*cs)) % q
                      for c in cs) for cs in product(range(q), repeat=3)]
        last = code
        md = min(sum(a != b for a, b in zip(code[i], code[j]))
                 for i in range(len(code)) for j in range(i + 1, len(code)))
        if md >= 4:
            return list(code)
    return last


def _ball_mask(q, n):
    mask = [set() for _ in range(q ** n)]
    for x in range(q ** n):
        s = {x}
        for i in range(n):
            for delta in range(1, q):
                s.add(x + delta * (q ** i))
        mask[x] = s
    return mask


def _min_vertex_cover(verts, edges):
    L = {v for v in verts if v % 2 == 0}
    R = verts - L
    adj = {v: set() for v in verts}
    for u, v in edges:
        if u in L and v in R:
            adj[u].add(v)
        elif v in L and u in R:
            adj[v].add(u)
    match = {v: None for v in verts}

    def bfs():
        dist = {v: 0 for v in L}
        queue = collections.deque(v for v in L if match[v] is None)
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                mu = match[v]
                if mu is None:
                    found = True
                elif dist.get(mu, 10 ** 9) == 10 ** 9:
                    dist[mu] = dist[u] + 1
                    queue.append(mu)
        return dist, found

    def dfs(u, dist, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            mu = match[v]
            if mu is None or (dist.get(mu, 10 ** 9) == dist[u] + 1 and dfs(mu, dist, seen)):
                match[v] = u
                match[u] = v
                return True
        return False

    while True:
        dist, found = bfs()
        if not found:
            break
        for u in L:
            if match[u] is None:
                dfs(u, dist, set())

    Z = set()
    stack = [u for u in L if match[u] is None]
    Z.update(stack)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in Z:
                Z.add(v)
                mu = match[v]
                if mu is not None and mu not in Z:
                    Z.add(mu)
                    stack.append(mu)
    return (L - Z) | (R & Z)


def _greedy_cover(words, mask):
    uncovered = set(range(len(words)))
    chosen = []
    while uncovered:
        best, bestc = None, -1
        for w in words:
            c = len(mask[w] & uncovered)
            if c > bestc:
                bestc, best = c, w
                if bestc == len(uncovered):
                    break
        chosen.append(best)
        uncovered -= mask[best]
    return chosen


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = t0 + 118.0
    total = q ** n
    random.seed(20240517)

    mask = _ball_mask(q, n)
    best = _greedy_cover(range(total), mask)
    best_len = len(best)

    code = _gen_code(q, n)
    code_set = set(code)
    cosets = [list(c) for c in code]
    coset_of = {tuple(c): i for i, c in enumerate(code)}

    for x in range(total):
        d = _to_tuple(x, q, n)
        ci = None
        for c in code:
            diff = tuple((d[i] - c[i]) % q for i in range(n))
            if diff in code_set:
                ci = coset_of[diff]
                break
        if ci is None:
            ci = len(cosets)
            cosets.append([d])
        else:
            cosets[ci].append(d)

    m = len(cosets)
    edges = set()
    for i in range(m):
        for j in range(i + 1, m):
            if sum(1 for a, b in zip(cosets[i][0], cosets[j][0]) if a != b) == 1:
                edges.add((i, j))

    cover = _min_vertex_cover(set(range(m)), edges)

    assigned = {}
    for ci in sorted(cover):
        bestw, bestc = None, -1
        for w in cosets[ci]:
            c = sum(1 for v in mask[w] if coset_of[_to_tuple(v, q, n)] not in assigned)
            if c > bestc:
                bestc, bestw = c, w
        assigned[ci] = bestw

    result = [assigned[ci] for ci in cover]
    if len(result) < best_len:
        best = result
        best_len = len(result)

    if time.time() < DEADLINE:
        chosen = list(best)
        for _ in range(4000):
            if time.time() > DEADLINE:
                break
            cov = set()
            for w in chosen:
                cov |= mask[w]
            improved = False
            for idx, w in enumerate(chosen):
                if cov - mask[w] == cov:
                    chosen.pop(idx)
                    improved = True
                    break
            if not improved:
                break
        if len(chosen) < best_len:
            best = chosen

    return [_to_tuple(w, q, n) for w in best]
# EVOLVE-BLOCK-END
