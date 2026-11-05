"""Rewritten: minimum vertex cover per graph (equivalent to hitting all maximal cliques size>=2)."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_vc(n, edges):
        # choose vertex covering the most uncovered edges, iteratively
        adj = build_adj(n, edges)
        deg = [len(adj[u]) for u in range(n)]
        covered = [False] * len(edges)
        chosen = []
        # map vertex -> incident edge indices
        inc = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            inc[u].append(i)
            inc[v].append(i)
        # active edges per vertex
        active = [set(inc[u]) for u in range(n)]
        for _ in range(n):
            # pick vertex with max uncovered incident edges
            best = -1
            bestc = -1
            for u in range(n):
                if not active[u]:
                    continue
                c = sum(1 for e in active[u] if not covered[e])
                if c > bestc:
                    bestc = c
                    best = u
            if best == -1 or bestc == 0:
                break
            chosen.append(best)
            for e in active[best]:
                if not covered[e]:
                    covered[e] = True
                    u, v = edges[e]
                    active[u].discard(e)
                    active[v].discard(e)
        return chosen

    def is_cover(n, edges, chosen):
        S = set(chosen)
        for u, v in edges:
            if u not in S and v not in S:
                return False
        return True

    def local_search(n, edges, chosen):
        S = set(chosen)
        improved = True
        while improved:
            improved = False
            for v in sorted(S):
                # try removing v
                S.remove(v)
                ok = True
                for u, w in edges:
                    if u == v or w == v:
                        other = w if u == v else u
                        if other not in S:
                            ok = False
                            break
                if ok:
                    chosen = list(S)
                    improved = True
                else:
                    S.add(v)
        return chosen

    def exact_vc(n, edges):
        # branch and bound with reduction
        adj = build_adj(n, edges)
        best = list(range(n))  # trivial upper bound
        # use a simple recursive B&B
        import sys
        sys.setrecursionlimit(10000)

        def solve(remaining_edges, current, depth):
            if not remaining_edges:
                if len(current) < len(best):
                    best[:] = current[:]
                return
            if len(current) >= len(best):
                return
            if time.time() > DEADLINE:
                return
            # pick edge with fewest options
            e = remaining_edges[0]
            u, v = e
            rest = remaining_edges[1:]
            # branch on u
            current.append(u)
            new_rest = [x for x in rest if u not in x]
            solve(new_rest, current, depth + 1)
            current.pop()
            # branch on v
            current.append(v)
            new_rest = [x for x in rest if v not in x]
            solve(new_rest, current, depth + 1)
            current.pop()

        # order edges by min degree to prune
        def order_edges(edge_list):
            if not edge_list:
                return []
            # pick edge with min (deg_u + deg_v)
            def key(e):
                a, b = e
                return len(adj[a]) + len(adj[b])
            i = min(range(len(edge_list)), key=lambda i: key(edge_list[i]))
            first = edge_list[i]
            rest = edge_list[:i] + edge_list[i+1:]
            return [first] + order_edges(rest)

        solve(order_edges(list(edges)), [], 0)
        return best

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # fallback: all vertices
            results.append(list(range(n)))
            continue
        edges = instance(n, p, seed)
        if not edges:
            results.append([])
            continue
        if n <= 32:
            # exact
            try:
                sol = exact_vc(n, edges)
            except Exception:
                sol = greedy_vc(n, edges)
            if not is_cover(n, edges, sol):
                sol = greedy_vc(n, edges)
        else:
            sol = greedy_vc(n, edges)
            if time.time() < DEADLINE - 5:
                sol = local_search(n, edges, sol)
            if not is_cover(n, edges, sol):
                sol = list(range(n))
        results.append(sorted(sol))
    return results
# EVOLVE-BLOCK-END
