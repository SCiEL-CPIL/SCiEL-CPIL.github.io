import time
import random

N = 15
_DEADLINE = 82.0  # safety margin under the 90s wall-clock limit


def _adj_and_deg(edges):
    adj = [set() for _ in range(N)]
    deg = [0] * N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg


def _is_valid(edges):
    # Reject if any simple 3-edge path s-u-v-t has deg(s) == deg(t).
    adj, deg = _adj_and_deg(edges)
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True


def _build_base():
    # Complete multipartite with distinct part sizes [1,2,3,4,5] (sum 15).
    # Distinct sizes => distinct degrees => constraint vacuously satisfied.
    sizes = [1, 2, 3, 4, 5]
    part_id = [0] * N
    v = 0
    for idx, s in enumerate(sizes):
        for _ in range(s):
            part_id[v] = idx
            v += 1
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if part_id[i] != part_id[j]:
                edges.append([i, j])
    return edges


def construct():
    start = time.time()
    best = _build_base()
    best_count = len(best)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    eset = frozenset(frozenset(e) for e in best)

    # Greedy add pass.
    for (i, j) in all_pairs:
        if time.time() - start > _DEADLINE:
            break
        if (i, j) in eset:
            continue
        trial = best + [[i, j]]
        if _is_valid(trial):
            best = trial
            eset = frozenset(frozenset(e) for e in best)
            best_count = len(best)

    # Time-guarded local search: add / remove / swap.
    random.seed(777)
    eset = frozenset(frozenset(e) for e in best)
    while time.time() - start < _DEADLINE:
        r = random.random()
        if r < 0.7:
            missing = [p for p in all_pairs if p not in eset]
            if not missing:
                break
            (i, j) = random.choice(missing)
            trial = best + [[i, j]]
            if _is_valid(trial):
                best = trial
                eset = frozenset(frozenset(e) for e in best)
                best_count = len(best)
        elif r < 0.85:
            if len(best) <= best_count:
                continue
            (i, j) = random.choice(best)
            best = [e for e in best if not (e[0] == i and e[1] == j)]
            eset = frozenset(frozenset(e) for e in best)
        else:
            if len(best) <= best_count:
                continue
            (a, b) = random.choice(best)
            missing = [p for p in all_pairs if p not in eset]
            if not missing:
                continue
            (c, d) = random.choice(missing)
            trial = [e for e in best if not (e[0] == a and e[1] == b)] + [[c, d]]
            if _is_valid(trial):
                best = trial
                eset = frozenset(frozenset(e) for e in best)
                best_count = len(best)

    # Final greedy polish.
    eset = frozenset(frozenset(e) for e in best)
    for (i, j) in all_pairs:
        if time.time() - start > _DEADLINE:
            break
        if (i, j) in eset:
            continue
        trial = best + [[i, j]]
        if _is_valid(trial):
            best = trial
            best_count = len(best)

    return [[int(u), int(v)] for u, v in best]
