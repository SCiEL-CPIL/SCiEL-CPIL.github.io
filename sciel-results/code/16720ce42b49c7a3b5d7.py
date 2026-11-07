"""Few cycle lengths under degree and girth constraints.

Strategy: build a triangle-free min-degree-3 graph on 12 vertices and
minimize the number of distinct simple-cycle lengths. We evaluate a set of
explicit constructions, then run a deadline-bounded local search that
mutates the edge set to reduce the distinct cycle-length count.
"""
# EVOLVE-BLOCK-START
import time
import random

N = 12
MIN_DEG = 3
DEADLINE = 55.0


def _now():
    return time.time()


def _adjacency(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _johnson_cycles(adj):
    n = len(adj)
    cycles = []

    def dfs(start, current, path, visited):
        for nb in sorted(adj[current]):
            if nb == start:
                cycles.append(len(path))
            elif nb > start and nb not in visited:
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, path, visited)
                path.pop()
                visited.discard(nb)

    for start in range(n):
        dfs(start, start, [start], {start})
    return cycles


def _distinct_lengths(edges):
    adj = _adjacency(edges)
    lens = _johnson_cycles(adj)
    return len(set(lens)) if lens else None


def _valid(edges):
    adj = _adjacency(edges)
    for i in range(N):
        if len(adj[i]) < MIN_DEG:
            return False
    for i in range(N):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj[i]:
                    return False
    lens = _johnson_cycles(adj)
    return bool(lens)


def _score(edges):
    if not _valid(edges):
        return None
    return _distinct_lengths(edges)


def _c6k2():
    e = set()
    for i in range(6):
        e.add((i, (i + 1) % 6))
        e.add((6 + i, 6 + (i + 1) % 6))
        e.add((i, 6 + i))
    return e


def _c4c3():
    e = set()
    for i in range(4):
        for j in range(3):
            u = i * 3 + j
            e.add((u, ((i + 1) % 4) * 3 + j))
            e.add((u, i * 3 + (j + 1) % 3))
    return e


def _c3c4():
    e = set()
    for i in range(3):
        for j in range(4):
            u = i * 4 + j
            e.add((u, ((i + 1) % 3) * 4 + j))
            e.add((u, i * 4 + (j + 1) % 4))
    return e


def _ring_squares(k):
    e = set()
    for j in range(k):
        u = 2 * j
        v = 2 * j + 1
        e.add((u, v))
        e.add((u, 2 * ((j + 1) % k)))
        e.add((v, 2 * ((j + 1) % k) + 1))
    return e


def _grid(a, b):
    e = set()
    for i in range(a):
        for j in range(b):
            u = i * b + j
            if j + 1 < b:
                e.add((u, u + 1))
            if i + 1 < a:
                e.add((u, u + b))
    return e


def _local_search(start_edges, deadline, seed):
    rng = random.Random(seed)
    best_edges = set(start_edges)
    best_score = _score(best_edges)
    if best_score is None:
        return None, None

    t0 = _now()
    cur = set(best_edges)
    cur_score = best_score
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    while _now() < deadline - 3.0:
        # propose a mutation: add or remove a random edge
        if rng.random() < 0.5 and len(cur) > 14:
            e = rng.choice(sorted(cur))
            cur.discard(e)
        else:
            e = rng.choice(all_pairs)
            cur.add(e)

        s = _score(cur)
        if s is None:
            # revert
            if e in cur:
                cur.discard(e)
            else:
                cur.add(e)
            continue
        # accept if better or equal (hill climbing); occasionally accept worse
        if s < cur_score or (s == cur_score and rng.random() < 0.3):
            cur_score = s
            if s < best_score:
                best_score = s
                best_edges = set(cur)
                if best_score == 1:
                    break
        else:
            # revert
            if e in cur:
                cur.discard(e)
            else:
                cur.add(e)
    return best_score, best_edges


def construct():
    t0 = _now()
    deadline = t0 + DEADLINE

    constructions = [
        _c6k2(),
        _c4c3(),
        _c3c4(),
        _ring_squares(6),
        _ring_squares(4),
        _grid(3, 4),
        _grid(4, 3),
        _grid(2, 6),
        _grid(6, 2),
    ]

    best_score = None
    best_edges = None
    for e in constructions:
        if _now() > deadline - 10.0:
            break
        s = _score(e)
        if s is not None and (best_score is None or s < best_score):
            best_score = s
            best_edges = set(e)
        if best_score == 1:
            break

    # local search from the best construction
    if best_edges is not None and best_score != 1 and _now() < deadline - 8.0:
        ls, le = _local_search(best_edges, deadline, seed=7)
        if ls is not None and (best_score is None or ls < best_score):
            best_score = ls
            best_edges = le

    if best_edges is None:
        best_edges = _c6k2()

    edges = sorted(tuple(sorted((u, v))) for u, v in best_edges)
    return [list(e) for e in edges]
# EVOLVE-BLOCK-END
