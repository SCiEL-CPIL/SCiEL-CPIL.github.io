# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0 - 6.0   # safety margin before the 150 s wall-clock kill


def _instance(n, d, seed):
    rng = random.Random(seed)
    degree, edges = [0] * n, set()
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edges or degree[u] >= d or degree[v] >= d:
            continue
        edges.add(e)
        degree[u] += 1
        degree[v] += 1
    return sorted(edges)


def _build(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # strong conflict: two edges conflict if they share a vertex or are joined by an edge
    conf = [set() for _ in range(len(edges))]
    for i in range(len(edges)):
        u, v = edges[i]
        for j in range(i + 1, len(edges)):
            a, b = edges[j]
            if (u == a or u == b or v == a or v == b
                    or a in adj[u] or b in adj[u] or a in adj[v] or b in adj[v]):
                conf[i].add(j)
                conf[j].add(i)
    return adj, conf


def _mergeable(cls, conf):
    ok = True
    for x, y in combinations(cls, 2):
        if y in conf[x]:
            ok = False
            break
    return ok


def _greedy_color(n, edges, conf):
    m = len(edges)
    order = sorted(range(m), key=lambda i: (-len(conf[i]), i))
    colour = [-1] * m
    for i in order:
        used = set()
        for j in conf[i]:
            c = colour[j]
            if c >= 0:
                used.add(c)
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    return colour


def _reassign(i, target, colour, conf):
    for j in conf[i]:
        if colour[j] == target:
            return False
    colour[i] = target
    return True


def _try_reassign(i, colour, conf, rng):
    for c in range(len(colour)):
        if c != colour[i] and _reassign(i, c, colour, conf):
            return True
    return False


def _local_search(colour, conf, rng, deadline):
    m = len(colour)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        order = list(range(m))
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline:
                break
            cur = colour[i]
            for c in range(cur):
                if _reassign(i, c, colour, conf):
                    improved = True
                    break
    return colour


def _merge_pass(colour, conf, rng, deadline):
    m = len(colour)
    maxc = max(colour) + 1
    changed = True
    while changed and time.time() < deadline:
        changed = False
        groups = [set() for _ in range(maxc)]
        for i, c in enumerate(colour):
            groups[c].add(i)
        order = list(range(maxc))
        rng.shuffle(order)
        for a in order:
            if not groups[a]:
                continue
            for b in order:
                if b <= a or not groups[b]:
                    continue
                if _mergeable(groups[a] | groups[b], conf):
                    for i in groups[b]:
                        colour[i] = a
                    groups[a] |= groups[b]
                    groups[b] = set()
                    changed = True
    return colour


def _solve_instance(n, d, seed, deadline):
    edges = _instance(n, d, seed)
    m = len(edges)
    if m == 0:
        return []
    adj, conf = _build(n, edges)
    colour = _greedy_color(n, edges, conf)
    rng = random.Random(1000 + seed)

    # initial reduction
    colour = _local_search(colour, conf, rng, deadline)
    colour = _merge_pass(colour, conf, rng, deadline)

    best = max(colour) + 1
    best_colour = colour[:]
    while time.time() < deadline:
        # simulated-annealing-style perturbation: reassign a random set of edges
        k = max(1, min(m, rng.randint(1, max(1, m // 20))))
        idxs = rng.sample(range(m), k)
        for i in idxs:
            if _try_reassign(i, best_colour, conf, rng):
                pass
        # re-optimize
        colour = _local_search(best_colour, conf, rng, deadline)
        colour = _merge_pass(colour, conf, rng, deadline)
        cur = max(colour) + 1
        if cur < best:
            best = cur
            best_colour = colour[:]
        # early stop at the conjecture bound
        if best <= 5 * d * d // 4:
            break
    return best_colour


def construct():
    deadline = _START + _DEADLINE
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    results = []
    for idx, (n, d, seed) in enumerate(specs):
        if time.time() > deadline:
            break
        results.append(_solve_instance(n, d, seed, deadline))
    return results
# EVOLVE-BLOCK-END
