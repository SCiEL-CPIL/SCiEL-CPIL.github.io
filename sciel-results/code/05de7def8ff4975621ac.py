"""Decompose three fixed random graphs into few edge-disjoint cycles and single edges."""
import time
import random
from collections import defaultdict, deque

_DEADLINE = 142.0  # safety margin below the 150s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() >= _DEADLINE


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle(adj, start):
    """Greedy path extension from `start` (commit, no backtrack)."""
    path = [start]
    parent = {start: None}
    cur = start
    while True:
        nxt = None
        for nb in sorted(adj[cur]):
            if nb not in parent:
                nxt = nb
                break
        if nxt is None:
            break
        parent[nxt] = cur
        path.append(nxt)
        cur = nxt
    if cur in parent and len(path) >= 3:
        # cur is adjacent to an earlier vertex -> close the cycle
        idx = path.index(cur)
        return path[idx:]
    return None


def _extract_2factor(adj):
    """Return a list of cycles (each a list of distinct vertices) covering as many
    vertices/edges as possible with a 2-regular spanning subgraph."""
    used = set()
    cycles = []
    for s in sorted(adj.keys()):
        if s in used:
            continue
        cyc = _find_cycle(adj, s)
        if cyc:
            used.update(cyc)
            cycles.append(cyc)
    # remove edges of these cycles from adj
    for cyc in cycles:
        m = len(cyc)
        for i in range(m):
            a, b = cyc[i], cyc[(i + 1) % m]
            adj[a].discard(b)
            adj[b].discard(a)
    return cycles


def _cycle_set(cyc):
    m = len(cyc)
    s = set()
    for i in range(m):
        a, b = cyc[i], cyc[(i + 1) % m]
        s.add((min(a, b), max(a, b)))
    return s


def _merge(cyc1, cyc2):
    """If two cycles share >=2 vertices, merge into one longer cycle (list)."""
    m1, m2 = len(cyc1), len(cyc2)
    e1 = _cycle_set(cyc1)
    e2 = _cycle_set(cyc2)
    union = e1 | e2
    shared = [v for v in cyc1 if v in set(cyc2)]
    if len(shared) < 2:
        return None
    # Build adjacency restricted to union edges; find a cycle through all vertices.
    adj = defaultdict(set)
    for a, b in union:
        adj[a].add(b)
        adj[b].add(a)
    verts = set(cyc1) | set(cyc2)
    if len(verts) < 3:
        return None
    # DFS to find a Hamiltonian cycle in this small 2-regular-ish graph.
    start = shared[0]
    found = None

    def dfs(node, path, used):
        nonlocal found
        if found is not None:
            return
        if len(path) == len(verts):
            if start in adj[node]:
                found = list(path)
            return
        for nb in sorted(adj[node]):
            if nb not in used and nb != start:
                used.add(nb)
                path.append(nb)
                dfs(nb, path, used)
                path.pop()
                used.discard(nb)
                if found is not None:
                    return
    dfs(start, [start], {start})
    return found


def _decompose(edges):
    adj = _build_adj(edges)
    pieces = []
    # Phase 1: repeatedly extract maximal 2-factors (cycle covers)
    while not _deadline_hit():
        prog = False
        for _ in range(200):
            if _deadline_hit():
                break
            cyc = _extract_2factor(adj)
            if not cyc:
                break
            prog = True
            pieces.extend(cyc)
        if not prog:
            break
    # Remaining edges are single-edge pieces
    for u, v in sorted(adj[u] for u in adj for v in adj[u]):
        pass
    remaining = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                remaining.add((u, v))
    pieces.extend([list(e) for e in sorted(remaining)])
    # Phase 2: merge cycles to reduce count
    if not _deadline_hit():
        changed = True
        while changed and not _deadline_hit():
            changed = False
            cycles = [p for p in pieces if len(p) >= 3]
            singles = [p for p in pieces if len(p) < 3]
            for i in range(len(cycles)):
                for j in range(i + 1, len(cycles)):
                    merged = _merge(cycles[i], cycles[j])
                    if merged is not None:
                        new_pieces = [p for k, p in enumerate(cycles) if k != i and k != j]
                        new_pieces.append(merged)
                        pieces = new_pieces + singles
                        changed = True
                        break
                if changed:
                    break
    return pieces


def _valid(pieces, edges):
    eset = {tuple(sorted(e)) for e in edges}
    seen = set()
    for p in pieces:
        if len(p) >= 3:
            m = len(p)
            if len(set(p)) != m:
                return False
            for i in range(m):
                a, b = p[i], p[(i + 1) % m]
                e = (min(a, b), max(a, b))
                if e in seen or e not in eset:
                    return False
                seen.add(e)
        else:
            e = tuple(sorted(p))
            if e in seen or e not in eset:
                return False
            seen.add(e)
    return seen == eset


def _score(pieces):
    return len(pieces)


def construct():
    best = None
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if _deadline_hit() and best is not None:
            break
        edges = _instance(*spec)
        result = _decompose(edges)
        if not _valid(result, edges):
            # fallback: every edge as its own piece
            result = [list(e) for e in edges]
        if best is None:
            best = [result]
        else:
            best.append(result)
    if best is None:
        # final safety fallback for all three
        best = [[list(e) for e in _instance(*spec)]
                for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))]
    return best
