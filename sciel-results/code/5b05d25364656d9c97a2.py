"""Rewritten program: edge-clique partition via repeated (near-)maximum clique extraction.

Approach family (new vs. seed): the seed lists every edge as its own 2-clique
(score = |E|).  Here we build a genuine *edge-clique cover*: we repeatedly find a
large clique in the residual graph, emit it, and delete its edges.  Because each
emitted clique absorbs O(|C|^2) edges at once, the number of cliques collapses from
|E| to a small multiple of |E| / (max-clique-size)^2.  The maximum-clique subproblem
is solved with the Tomita / MCQ branch-and-bound with greedy-coloring upper bound on
bitsets, which is fast for n <= 64.  The result is always a valid partition (every
edge is removed exactly once, and every emitted set is a clique of the original
graph) with at least two vertices.
"""

# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def lsb_index(x):
        return (x & -x).bit_length() - 1

    def build_edges(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def max_clique(adj, n, node_limit=4_000_000):
        """Return a (near-)maximum clique of the graph given by bitset adjacency `adj`."""
        best = []
        counter = [0]
        aborted = [False]

        def color_sort(P):
            order = []
            Pset = P
            color = 0
            while Pset:
                color += 1
                U = Pset
                while U:
                    v = lsb_index(U)
                    U &= U - 1
                    Pset &= ~(1 << v)
                    order.append((v, color))
            return order

        def expand(R, P):
            if aborted[0]:
                return
            counter[0] += 1
            if counter[0] > node_limit:
                aborted[0] = True
                return
            if not P:
                if len(R) > len(best):
                    best = list(R)
                return
            order = color_sort(P)
            for v, c in reversed(order):
                if aborted[0]:
                    return
                if len(R) + c <= len(best):
                    return
                R.append(v)
                newP = P & adj[v] & ((1 << v) - 1)
                expand(R, newP)
                R.pop()
                P &= ~(1 << v)

        expand([], (1 << n) - 1)
        return best

    def greedy_large_clique(adj, n):
        """Fast fallback: a reasonably large clique via degree-greedy construction."""
        deg = [adj[v].bit_count() for v in range(n)]
        if max(deg) < 1:
            return []
        # start from the highest-degree vertex
        start = max(range(n), key=lambda v: deg[v])
        clique = [start]
        cand = adj[start]
        while cand:
            # pick candidate with max degree within cand
            best_v = -1
            best_d = -1
            U = cand
            while U:
                v = lsb_index(U)
                d = (adj[v] & cand).bit_count()
                if d > best_d:
                    best_d = d
                    best_v = v
                U &= U - 1
            if best_v < 0:
                break
            clique.append(best_v)
            cand &= adj[best_v]
        return clique

    def solve_instance(n, p, seed):
        edges = build_edges(n, p, seed)
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u

        cliques = []
        # safety: never loop forever
        for _ in range(100000):
            # check if any edge remains
            any_edge = False
            for v in range(n):
                if adj[v]:
                    any_edge = True
                    break
            if not any_edge:
                break

            C = max_clique(adj, n)
            if len(C) < 2:
                # fallback to a single edge (guaranteed to exist)
                C = None
                for u in range(n):
                    if adj[u]:
                        v = lsb_index(adj[u])
                        C = [u, v]
                        break
            if C is None or len(C) < 2:
                # extremely defensive: take first available edge
                for u in range(n):
                    if adj[u]:
                        v = lsb_index(adj[u])
                        C = [u, v]
                        break
            cliques.append(C)
            # delete all edges inside C from the residual graph
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        return cliques

    return [
        solve_instance(32, 0.6, 601),
        solve_instance(64, 0.55, 602),
    ]
# EVOLVE-BLOCK-END

