"""Decompose each instance graph into edge-disjoint cycles + paths + edges, minimizing pieces."""
import time

_DEADLINE = 145.0  # safety margin under the 150s kill

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    if not edges:
        return []
    edges = set(edges)
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def find_cycle(start):
        # Greedy DFS: prefer neighbours that are "ahead" in the order (to close cycles)
        # and low-degree (to keep high-degree hubs available for more cycles).
        stack = [start]
        path = [start]
        on_path = {start}
        parent = {start: None}
        best = None
        while stack:
            v = stack[-1]
            nbrs = adj[v]
            # choose next neighbour
            nxt = None
            for w in nbrs:
                if (v, w) not in edges and (w, v) not in edges:
                    continue
                if w == path[0]:
                    if len(path) >= 3:
                        best = path[:]
                        return best
                    continue
                if w not in on_path:
                    nxt = w
                    break
            if nxt is None:
                stack.pop()
                on_path.discard(v)
                continue
            # commit
            parent[nxt] = v
            on_path.add(nxt)
            path.append(nxt)
            stack.append(nxt)
        return best

    def find_path(start, max_len=2):
        # find a simple path of exactly max_len edges (max_len+1 vertices)
        # returns list of vertices or None
        if max_len == 1:
            # any single remaining edge
            for (a, b) in list(edges):
                return [a, b]
            return None
        # max_len == 2: find a-b-c with both edges present
        for a in list(adj.keys()):
            for b in adj[a]:
                for c in adj[b]:
                    if c == a:
                        continue
                    e1 = (a, b) if (a, b) in edges else (b, a)
                    e2 = (b, c) if (b, c) in edges else (c, b)
                    if e1 in edges and e2 in edges and e1 != e2:
                        return [a, b, c]
        return None

    def remove_piece(piece):
        # piece: list of vertices (cycle closes, path/edge doesn't)
        if len(piece) >= 3:
            for i in range(len(piece)):
                u = piece[i]
                v = piece[(i + 1) % len(piece)]
                e = (u, v) if (u, v) in edges else (v, u)
                edges.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            u, v = piece
            e = (u, v) if (u, v) in edges else (v, u)
            edges.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)

    pieces = []
    # Phase 1: extract as many cycles as possible
    while edges:
        if time.time() > deadline:
            break
        # pick a start vertex with highest degree among remaining
        start = None
        best_deg = -1
        for v in adj:
            d = len(adj[v])
            if d >= 2 and d > best_deg:
                best_deg = d
                start = v
        if start is None:
            break
        cyc = find_cycle(start)
        if cyc is not None:
            pieces.append(cyc)
            remove_piece(cyc)
        else:
            # no more cycles reachable from any high-degree start; break to path phase
            break

    # Phase 2: extract 2-edge paths while possible
    while edges and time.time() <= deadline:
        p2 = find_path(0, 2)
        if p2 is None:
            break
        pieces.append(p2)
        remove_piece(p2)

    # Phase 3: remaining single edges
    for (a, b) in list(edges):
        pieces.append([a, b])
        remove_piece([a, b])

    return pieces


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        pieces = _decompose(edges, deadline)
        results.append(pieces)
    return results
