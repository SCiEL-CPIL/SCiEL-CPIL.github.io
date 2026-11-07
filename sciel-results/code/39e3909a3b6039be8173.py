"""Structural clique-partition exploiting split-graph structure: C is a clique, I is independent.
Each clique in the partition is either a subset of C, or a subset of C plus at most one vertex from I.
Optimal: one clique = all of C (covers all C-C edges), then for each I-vertex v with deg(v)>=1,
one clique = N_C(v) + {v} (covers all cross edges incident to v). Total = 1 + |{v in I: deg(v)>=1}|."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    _T0 = time.time()
    _DEADLINE = 150.0 - 12.0  # safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_one(n, k, p, seed):
        # Build adjacency
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))            # clique side
        I = list(range(k, n))         # independent side

        cliques = []

        # One big clique for all of C (covers all internal C-C edges)
        if k >= 2:
            cliques.append(list(C))

        # For each I-vertex v with at least one cross edge, add clique N_C(v) + {v}
        for v in I:
            nbrs_C = [u for u in C if u in adj[v]]
            if len(nbrs_C) >= 1:
                cliques.append(sorted(nbrs_C + [v]))

        # Validate
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))

        def _is_clique(verts):
            vs = list(verts)
            for i in range(len(vs)):
                a = vs[i]
                for j in range(i + 1, len(vs)):
                    if vs[j] not in adj[a]:
                        return False
            return True

        for cl in cliques:
            assert len(cl) >= 2, f"clique too small: {cl}"
            assert len(set(cl)) == len(cl), f"duplicate vertices: {cl}"
            assert _is_clique(cl), f"not a clique: {cl}"

        seen = set()
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (cl[a], cl[b])
                    assert e not in seen, f"edge {e} in two cliques"
                    assert e in all_edges, f"edge {e} not in graph"
                    seen.add(e)
        assert seen == all_edges, f"missing edges: {all_edges - seen}"

        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            # fallback: one clique per edge
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
