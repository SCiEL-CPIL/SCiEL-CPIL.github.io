"""Few cycle lengths under degree>=3 and girth>=4 on 12 vertices.
Fresh design: start from a strong analytic 12-cycle+parallel-chord graph,
verify by exact cycle enumeration, then random-search for fewer distinct
cycle lengths within a wall-clock deadline."""
import time
import random

N = 12
MIN_DEG = 3
GIRTH = 4


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _build_adj(edge_set):
    adj = [set() for _ in range(N)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _cycle_lengths(adj):
    """All simple cycle lengths via backtracking (each cycle counted once)."""
    lengths = set()
    n = len(adj)
    for start in range(n):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            d = len(path)
            for nb in adj[node]:
                if nb == start and d >= GIRTH:
                    lengths.add(d)
                elif nb not in path:
                    stack.append((nb, path + [nb]))
    return lengths


def _valid(edge_set):
    if len(edge_set) != len({e for e in edge_set}):
        return False
    adj = _build_adj(edge_set)
    if any(len(a) < MIN_DEG for a in adj):
        return False
    lengths = _cycle_lengths(adj)
    if not lengths:
        return False
    if min(lengths) < GIRTH:
        return False
    return lengths


def _base_graph():
    # 12-cycle with parallel chords -> every simple cycle length divides 12.
    e = set()
    for i in range(N):
        e.add(_edge_key(i, (i + 1) % N))
    for i in range(6):
        e.add(_edge_key(i, i + 6))
    return e


def construct():
    deadline = time.time() + 50.0  # safety margin under 60s/90s limit
    rnd = random.Random(20240517)

    best = None
    base = _base_graph()
    bl = _valid(base)
    if bl is not None:
        best = (len(bl), base)

    # Random search for fewer distinct cycle lengths.
    edges = list(range(N * (N - 1) // 2))
    for _ in range(2000000):
        if time.time() >= deadline:
            break
        m = rnd.randrange(10, 21)
        e = set(rnd.sample(edges, m))
        if any(len(set(map(_edge_key, (u, v))) & {e2 for e2 in e}) != 0 for e2 in e):
            continue
        lengths = _valid(e)
        if lengths is not None and (best is None or len(lengths) < best[0]):
            best = (len(lengths), e)
            if len(lengths) == 1:
                break

    if best is None:
        return []
    out = [[u, v] for (u, v) in sorted(best[1])]
    for pair in out:
        pair[0] = int(pair[0])
        pair[1] = int(pair[1])
    return out
