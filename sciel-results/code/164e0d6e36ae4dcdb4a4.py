"""Strong edge colouring via branch-and-bound on the conflict graph."""
import time
import sys

sys.setrecursionlimit(100000)

_DEADLINE = 145.0
_START = time.time()


def construct():
    from random import Random

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        m = len(edges)
        # near[v] = {v} ∪ N(v)
        near = [None] * n
        for v in range(n):
            near[v] = set(adj[v])
            near[v].add(v)
        # conflict adjacency over edge-indices
        cadj = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                a, b = edges[j]
                # share a vertex
                if a == u or a == v or b == u or b == v:
                    cadj[i].add(j)
                    cadj[j].add(i)
                    continue
                # joined by an edge
                if a in near[u] or a in near[v] or b in near[u] or b in near[v]:
                    cadj[i].add(j)
                    cadj[j].add(i)
        return cadj

    def greedy_color(cadj, m, order=None):
        if order is None:
            order = sorted(range(m), key=lambda v: -len(cadj[v]))
        color = [-1] * m
        for v in order:
            used = set()
            for nb in cadj[v]:
                if color[nb] != -1:
                    used.add(color[nb])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        return color, (max(color) + 1) if m else 0

    def color_bnb(cadj, m, ub, deadline):
        # exact branch-and-bound to minimize colours
        # global best
        best = [ub]
        best_color = [None]
        # order: most constrained first
        order = sorted(range(m), key=lambda v: -len(cadj[v]))
        pos = {v: i for i, v in enumerate(order)}
        # For each vertex, precompute its neighbours that come AFTER it in order
        # (so when we colour order[i], all later neighbours are uncolored)
        # We'll just check all neighbours.

        color = [-1] * m
        # clique lower bound helper on a subset (greedy clique)
        # We compute a quick lower bound = max clique found greedily among uncolored
        # but recomputing is expensive; instead use: number of uncolored vertices
        # that are pairwise... simpler: use the max degree among uncolored + 1 as lb
        # Actually a solid lb: size of a clique in the remaining graph.
        # We approximate with a greedy clique on the uncolored induced subgraph.

        def clique_lb(uncolored):
            # greedy maximal clique, returns its size (a valid lower bound)
            if not uncolored:
                return 0
            rem = set(uncolored)
            size = 0
            while rem:
                # pick vertex with max degree within rem
                v = max(rem, key=lambda x: len(cadj[x] & rem))
                size += 1
                rem = rem & cadj[v]
            return size

        # track uncolored count
        uncolored = set(range(m))

        def dfs(idx):
            nonlocal best, best_color
            if time.time() > deadline:
                return False  # timeout signal
            if idx == m:
                # all colored; best_color is implicit in color
                return True
            v = order[idx]
            # lower bound: current used colors + clique lb of uncolored (incl. v)
            # used colors so far
            used_colors = 0
            # compute max color used among colored
            max_used = -1
            for nb in range(m):
                if color[nb] != -1:
                    if color[nb] > max_used:
                        max_used = color[nb]
            # lower bound on final = max(max_used+1, clique_lb(uncolored))
            lb = clique_lb(uncolored)
            if max_used + 1 > lb:
                lb = max_used + 1
            if lb >= best[0]:
                return True  # prune, but not timeout
            # try colors 0..lb-1 (and maybe up to lb)
            # forbidden by neighbors
            for c in range(lb):
                ok = True
                for nb in cadj[v]:
                    if color[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                color[v] = c
                uncolored.discard(v)
                if dfs(idx + 1):
                    if time.time() > deadline:
                        return False
                    if all(col != -1 for col in color):
                        # found a full coloring with <= c+1 colors
                        pass
                    # if we completed, record
                    # detect completion
                    if idx + 1 == m:
                        final = max(color) + 1
                        if final < best[0]:
                            best[0] = final
                            best_color[0] = color[:]
                    return True
                color[v] = -1
                uncolored.add(v)
            # if no color worked under lb, we may need color == lb
            # try color = lb (one extra)
            c = lb
            ok = True
            for nb in cadj[v]:
                if color[nb] == c:
                    ok = False
                    break
            if ok:
                color[v] = c
                uncolored.discard(v)
                if dfs(idx + 1):
                    if time.time() > deadline:
                        return False
                    if idx + 1 == m:
                        final = max(color) + 1
                        if final < best[0]:
                            best[0] = final
                            best_color[0] = color[:]
                    return True
                color[v] = -1
                uncolored.add(v)
            return True

        dfs(0)
        return best[0], best_color[0]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue
        cadj = build_conflict(edges, spec[0])
        # greedy upper bound
        gcol, ub = greedy_color(cadj, m)
        deadline = _DEADLINE - 3.0  # safety margin
        if time.time() > deadline:
            out.append(gcol)
            continue
        # try exact; if it finishes, use it; else fall back to greedy
        try:
            bnb_col, bnb_col[0] if False else None  # placeholder
        except Exception:
            bnb_col = None
        # call bnb
        try:
            best_k, best_assign = color_bnb(cadj, m, ub, deadline)
            if best_assign is not None:
                out.append(best_assign)
            else:
                out.append(gcol)
        except Exception:
            out.append(gcol)
        if time.time() > _DEADLINE - 5.0:
            # not enough time left; fill remaining with greedy quickly
            for spec2 in ((40, 4, 101), (60, 5, 102), (80, 6, 103))[len(out):]:
                e2 = instance(*spec2)
                if not e2:
                    out.append([])
                    continue
                ca2 = build_conflict(e2, spec2[0])
                gc2, _ = greedy_color(ca2, len(e2))
                out.append(gc2)
            break
    return out
