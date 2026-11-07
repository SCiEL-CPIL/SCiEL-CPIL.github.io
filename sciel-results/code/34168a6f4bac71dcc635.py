"""Rewritten program: clique edge-partition of split graphs."""
import time

def construct():
    start = time.time()
    deadline = start + 140.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique(V, adj):
        V = set(V)
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            pool = P | X
            u = max(pool, key=lambda x: len(adj[x] & P)) if pool else None
            for v in list(P - (adj[u] if u is not None else set())):
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.remove(v)
                P.remove(v)
                X.add(v)
        bk(set(), V, set())
        return best

    def solve(n, k, p, seed):
        K = list(range(k))
        I = list(range(k, n))
        kadj = {u: set() for u in K}
        for i in range(k):
            for j in range(i + 1, k):
                kadj[i].add(j); kadj[j].add(i)
        rng = Random(seed)
        cross = {}
        for v in I:
            nbrs = set()
            for u in K:
                if rng.random() < p:
                    nbrs.add(u)
            cross[v] = nbrs

        cliques = []
        covered_KE = set()  # K-K edges covered by cross cliques

        # Cross cliques: for each v, partition neighbors into max cliques of K-graph
        for v in I:
            remaining = set(cross[v])
            while remaining:
                adj_r = {x: (kadj[x] & remaining) for x in remaining}
                mc = max_clique(remaining, adj_r)
                if not mc:
                    mc = [next(iter(remaining))]
                cliques.append(sorted(mc) + [v])
                # mark K-K edges inside mc as covered
                mcl = sorted(mc)
                for a in range(len(mcl)):
                    for b in range(a + 1, len(mcl)):
                        covered_KE.add((mcl[a], mcl[b]))
                remaining -= set(mc)

        # Cover remaining K-K edges with dedicated cliques (K is a full clique,
        # so any subset is a clique). Group remaining edges into cliques.
        rem_edges = []
        for i in range(k):
            for j in range(i + 1, k):
                if (i, j) not in covered_KE:
                    rem_edges.append((i, j))
        # Greedy: build cliques covering remaining edges. Since K is a clique,
        # a clique on a set S covers all edges within S. Greedily pick a vertex,
        # extend with vertices sharing uncovered edges.
        remaining_edges = set(rem_edges)
        while remaining_edges:
            # pick a vertex incident to many remaining edges
            e0 = next(iter(remaining_edges))
            clique = set([e0[0], e0[1]])
            # grow: add vertex w if it forms a clique (always, K is clique) 
            # and covers at least one remaining edge with current clique
            changed = True
            while changed:
                changed = False
                for w in K:
                    if w in clique:
                        continue
                    # does w share a remaining edge with any in clique?
                    for x in clique:
                        if (min(w, x), max(w, x)) in remaining_edges:
                            clique.add(w)
                            changed = True
                            break
            cl = sorted(clique)
            # remove all edges within cl that are still remaining
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    remaining_edges.discard((cl[a], cl[b]))
            if len(cl) >= 2:
                cliques.append(cl)

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        result.append(solve(n, k, p, seed))

    cleaned = [[[int(x) for x in c] for c in clist] for clist in result]
    return cleaned
