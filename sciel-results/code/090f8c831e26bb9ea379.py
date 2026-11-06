"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
import time
import random as _random

# EVOLVE-BLOCK-START

_DEADLINE = 150.0
_MARGIN = 15.0


def construct():
    t0 = time.time()
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = _gen_edges(n, p, seed)
        adj = {}
        for (u, v) in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        pieces = []
        edge_set = set(edges)
        while edge_set:
            if time.time() - t0 > _DEADLINE - _MARGIN:
                # Fallback: remaining edges as single-edge pieces
                for e in edge_set:
                    pieces.append([e[0], e[1]])
                break
            # Try to find a long cycle; else take a single edge
            cyc = _find_long_cycle(adj, edge_set)
            if cyc and len(cyc) >= 3:
                pieces.append(cyc)
                for i in range(len(cyc)):
                    a = cyc[i]
                    b = cyc[(i + 1) % len(cyc)]
                    edge_set.discard(tuple(sorted((a, b))))
            else:
                e = next(iter(edge_set))
                pieces.append([e[0], e[1]])
                edge_set.discard(e)
        results.append(pieces)
    return results


def _gen_edges(n, p, seed):
    rng = _random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_long_cycle(adj, edge_set):
    # Greedy longest-path-based cycle search with a few random restarts
    best = None
    # Candidate start vertices: those with degree >= 2
    starts = [v for v in adj if len(adj[v]) >= 2]
    if not starts:
        return None
    # Deterministic first, then a few random restarts
    order = list(starts)
    for s in order[:1] + (order[1:4] if len(order) > 1 else []):
        cyc = _longest_cycle_from(adj, edge_set, s)
        if cyc and (best is None or len(cyc) > len(best)):
            best = cyc
    return best


def _longest_cycle_from(adj, edge_set, start):
    # DFS to find a cycle through start, maximizing length
    best = [None]
    parent = {start: None}
    visited = {start}

    def dfs(u, path):
        if len(path) >= 3 and start in adj.get(u, ()):
            # closing edge (u, start)
            e = tuple(sorted((u, start)))
            if e in edge_set:
                cyc = path[:]
                if best[0] is None or len(cyc) > len(best[0]):
                    best[0] = cyc[:]
        for v in sorted(adj.get(u, ())):
            if v in visited:
                continue
            e = tuple(sorted((u, v)))
            if e not in edge_set:
                continue
            visited.add(v)
            parent[v] = u
            path.append(v)
            dfs(v, path)
            path.pop()
            visited.discard(v)
            parent.pop(v, None)

    dfs(start, [start])
    return best[0]


# EVOLVE-BLOCK-END
