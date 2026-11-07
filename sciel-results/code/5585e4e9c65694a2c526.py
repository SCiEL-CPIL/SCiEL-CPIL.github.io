"""Rewritten: minimum clique partition of split graphs using structural cliques."""
import time

# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _norm(e):
        return (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(_norm((u, v)))

        K = list(range(k))
        S = list(range(k, n))

        assigned = set()
        cliques = []

        # Structural cliques: for each S-vertex w with cross edges,
        # C_w = {w} ∪ N_K(w) is a clique covering all cross edges of w
        # plus K-edges inside N_K(w).
        for w in S:
            Nw = [u for u in K if u in adj[w]]
            if not Nw:
                continue
            C = [w] + Nw
            cl = set()
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    cl.add(_norm((C[i], C[j])))
            new_edges = [e for e in cl if e not in assigned]
            if new_edges:
                cliques.append(sorted(C))
                assigned.update(new_edges)

        # Leftover edges (a subgraph of K). Cover with greedy max-clique packing.
        leftover = set(e for e in edge_set if e not in assigned)
        deadline = time.time() + 100.0
        while leftover:
            if time.time() > deadline:
                for e in leftover:
                    cliques.append(sorted(e))
                break
            start = next(iter(leftover))
            C = {start[0], start[1]}
            changed = True
            while changed:
                changed = False
                inter = None
                for x in C:
                    inter = set(adj[x]) if inter is None else (inter & adj[x])
                inter = inter - C
                for w in inter:
                    ok = True
                    for x in C:
                        if _norm((w, x)) not in leftover:
                            ok = False
                            break
                    if ok:
                        C.add(w)
                        changed = True
                        break
            cl = set()
            lst = list(C)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    e = _norm((lst[i], lst[j]))
                    if e in leftover:
                        cl.add(e)
            if cl:
                cliques.append(sorted(C))
                leftover -= cl
            else:
                # safety: take a single edge
                e = next(iter(leftover))
                cliques.append(sorted(e))
                leftover.discard(e)

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        clqs = solve(n, k, p, seed)
        clqs = [[int(v) for v in c] for c in clqs]
        result.append(clqs)
    return result
# EVOLVE-BLOCK-END
