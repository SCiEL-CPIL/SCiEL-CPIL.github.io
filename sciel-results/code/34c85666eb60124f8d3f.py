"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Build adjacency with edge ids for tracking
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = {}
        for u, v in edge_set:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        # remaining unused edges
        remaining = set(edge_set)
        # degree of remaining edges per vertex
        rem_deg = {v: 0 for v in adj}
        for u, v in remaining:
            rem_deg[u] += 1
            rem_deg[v] += 1

        paths = []

        def try_extend(path, end, other_end):
            # Extend 'path' at the 'end' vertex using an unused edge.
            # Returns True if extended.
            end_v = path[end]
            # candidate neighbors with unused edge
            cands = [w for w in adj.get(end_v, ())
                     if (min(end_v, w), max(end_v, w)) in remaining
                     and w not in path]
            if not cands:
                return False
            # prefer neighbor with smallest remaining degree (tightness heuristic)
            cands.sort(key=lambda w: (rem_deg[w], w))
            w = cands[0]
            # commit
            if end == 0:
                path.insert(0, w)
            else:
                path.append(w)
            e = (min(end_v, w), max(end_v, w))
            remaining.discard(e)
            rem_deg[end_v] -= 1
            rem_deg[w] -= 1
            return True

        while remaining:
            # pick a starting edge: one whose endpoints have smallest remaining degree
            start_edge = min(remaining, key=lambda e: (rem_deg[e[0]] + rem_deg[e[1]], e))
            u, v = start_edge
            path = [u, v]
            remaining.discard((u, v))
            rem_deg[u] -= 1
            rem_deg[v] -= 1
            # extend both ends greedily
            progress = True
            while progress:
                progress = False
                # extend left
                if try_extend(path, 0, 1):
                    progress = True
                # extend right
                if try_extend(path, 1, 0):
                    progress = True
            paths.append(path)

        # Verify coverage and simplicity (defensive)
        used = set()
        for p in paths:
            assert len(p) == len(set(p)), "path not simple"
            for a, b in zip(p, p[1:]):
                e = (min(a, b), max(a, b))
                assert e in edge_set, "edge not in graph"
                assert e not in used, "edge reused"
                used.add(e)
        assert used == edge_set, "not all edges covered"
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
# EVOLVE-BLOCK-END
