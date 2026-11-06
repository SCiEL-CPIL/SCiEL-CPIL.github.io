import time
import random
import itertools

# EVOLVE-BLOCK-START
def _build_adj(edge_list):
    adj = [set() for _ in range(18)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_k4(adj):
    # Fast K4 detector: for each 3-clique, check for a common neighbor.
    n = 18
    for a in range(n):
        for b in range(a + 1, n):
            if b not in adj[a]:
                continue
            common_ab = adj[a] & adj[b]
            for c in common_ab:
                if c <= b:
                    continue
                if c not in adj[a] or c not in adj[b]:
                    continue
                # a,b,c is a triangle; K4 exists if they share a common neighbor
                if (adj[a] & adj[b] & adj[c]):
                    return True
    return False

def _is_k4_free(edge_list):
    return not _has_k4(_build_adj(edge_list))

def _largest_triangle_free(adj):
    n = 18
    # exact branch and bound over independent sets (triangle-free <=> independent in G^2)
    sq = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            sq[i].add(j)
            sq[j].add(i)
    # degree ordering for branching
    order = sorted(range(n), key=lambda v: len(sq[v]), reverse=True)
    pos = {v: i for i, v in enumerate(order)}
    best = [0]

    def rec(cand, depth, size):
        if depth + len(cand) <= best[0]:
            return
        if not cand:
            if size > best[0]:
                best[0] = size
            return
        # branch on the first candidate in order
        v = cand[0]
        rest = cand[1:]
        # take v
        newcand = [x for x in rest if x not in sq[v]]
        rec(newcand, depth + 1, size + 1)
        # skip v
        rec(rest, depth, size)

    rec(order, 0, 0)
    return best[0]

def _score(edge_list):
    adj = _build_adj(edge_list)
    return _largest_triangle_free(adj)

def _complete_multipartite(parts):
    edges = []
    k = len(parts)
    for i in range(k):
        for j in range(i + 1, k):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges

def _random_partition(seed):
    rng = random.Random(seed)
    parts = [[i] for i in range(18)]
    rng.shuffle(parts)
    return parts

def _local_search(edge_list, seed, deadline):
    rng = random.Random(seed)
    adj = _build_adj(edge_list)
    cur = _largest_triangle_free(adj)
    best = cur
    best_edges = [list(e) for e in edge_list]
    all_pairs = [(i, j) for i in range(18) for j in range(i + 1, 18)]
    while time.time() < deadline:
        # pick a random edge toggle
        u, v = rng.choice(all_pairs)
        if v in adj[u]:
            adj[u].discard(v)
            adj[v].discard(u)
        else:
            adj[u].add(v)
            adj[v].add(u)
        if _has_k4(adj):
            # revert
            if v in adj[u]:
                adj[u].discard(v)
                adj[v].discard(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
            continue
        s = _largest_triangle_free(adj)
        if s <= cur:
            cur = s
            if s < best:
                best = s
                best_edges = [list(e) for e in _edges_from_adj(adj)]
        else:
            # revert
            if v in adj[u]:
                adj[u].discard(v)
                adj[v].discard(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
    return best, best_edges

def _edges_from_adj(adj):
    return [[i, j] for i in range(18) for j in range(i + 1, 18) if j in adj[i]]

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill
    rng = random.Random(12345)

    # candidate pool: structural priors + random partitions
    candidates = []
    # balanced complete multipartite
    for k in range(2, 18):
        sizes = [18 // k] * k
        for i in range(18 % k):
            sizes[i] += 1
        parts = []
        idx = 0
        for s in sizes:
            parts.append(list(range(idx, idx + s)))
            idx += s
        candidates.append(_complete_multipartite(parts))
    # random partitions (few)
    for seed in range(40):
        parts = _random_partition(seed)
        candidates.append(_complete_multipartite(parts))

    best = 18
    best_edges = None
    for edges in candidates:
        if time.time() > deadline:
            break
        if not _is_k4_free(edges):
            continue
        s = _score(edges)
        if s < best:
            best = s
            best_edges = [list(e) for e in edges]
        if best <= 8:
            break

    # local search from the best found
    if best_edges is not None:
        b, be = _local_search(best_edges, 999, deadline)
        if b < best:
            best = b
            best_edges = be

    # If nothing worked, fall back to a known K4-free graph
    if best_edges is None:
        best_edges = _complete_multipartite([list(range(0, 9)), list(range(9, 18))])
        best = _score(best_edges)

    # final validity guard
    if not _is_k4_free(best_edges):
        best_edges = _complete_multipartite([list(range(0, 9)), list(range(9, 18))])
        best = _score(best_edges)

    return best_edges
# EVOLVE-BLOCK-END
