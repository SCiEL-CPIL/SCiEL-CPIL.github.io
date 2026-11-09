"""C4-free graph with high minimum degree: greedy edge removal from K_n."""
import time
import random

_START = time.time()
_DEADLINE = 110.0  # safety margin before the 120s kill


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _valid(n, edges):
    """Check simple graph, no C4, and min-degree >= n-39."""
    adj = [set() for _ in range(n)]
    for a, b in edges:
        if a == b:
            return False
        if b in adj[a]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    for v in range(n):
        if len(adj[v]) < n - 39:
            return False
    # C4 check via shared-neighbor pairs
    for v in range(n):
        nb = list(adj[v])
        for i in range(len(nb)):
            a = nb[i]
            for j in range(i + 1, len(nb)):
                b = nb[j]
                if len(adj[a] & adj[b]) > 0:
                    return False
    return True


def _build(n, seed):
    """Random greedy removal: start from K_n, delete edges to kill every C4."""
    rng = random.Random(seed)
    adj = [set(range(n)) for _ in range(n)]
    for v in range(n):
        adj[v].discard(v)

    # All pairs of shared neighbors of a vertex = C4 witnesses
    witnesses = {}
    for v in range(n):
        nb = list(adj[v])
        for i in range(len(nb)):
            a = nb[i]
            for j in range(i + 1, len(nb)):
                b = nb[j]
                if a > b:
                    a, b = b, a
                witnesses.setdefault((a, b), set()).add(v)

    # Priority: edges that witness the most C4s first
    order = sorted(witnesses.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    for (a, b), ws in order:
        if not ws:
            continue
        # Remove edge only if it still witnesses some C4
        for v in ws:
            if a in adj[v] and b in adj[v]:
                adj[a].discard(b)
                adj[b].discard(a)
                break

    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges


def _try_n(n, seed):
    edges = _build(n, seed)
    return edges if _valid(n, edges) else None


def construct():
    best = {'n': 39, 'edges': []}
    # Try large n first; multiple seeds to increase success probability
    for n in (47, 46, 45, 44, 43, 42):
        if _remaining() < 5.0:
            break
        found = None
        for seed in range(200000):
            if _remaining() < 2.0:
                break
            res = _try_n(n, seed)
            if res is not None:
                found = res
                break
        if found is not None:
            best = {'n': n, 'edges': found}
            break
    return best
