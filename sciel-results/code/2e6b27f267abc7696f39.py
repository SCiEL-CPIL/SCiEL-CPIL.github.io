import time
import random
import heapq

_GLOBAL_START = time.time()
_GLOBAL_BUDGET = 140.0  # safety margin under the 150 s wall-clock kill

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def make_cover_matrix(cliques, n):
        # cover[v] = frozenset of clique indices that contain vertex v
        cover = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                cover[v].add(ci)
        return [frozenset(s) for s in cover]

    def max_independent_set_size(m, share, limit=2000000):
        # Maximum independent set in the 'clique-intersection graph' (m nodes,
        # share[i][j] = True iff cliques i,j share a vertex). An independent set
        # is a set of pairwise-disjoint cliques. Computed via Bron-Kerbosch on
        # the complement graph. Returns (size, solved_bool).
        if m <= 1:
            return m, True
        # complement adjacency: not share
        comp = [set() for _ in range(m)]
        for i in range(m):
            si = comp[i]
            for j in range(m):
                if i != j and not share[i][j]:
                    si.add(j)
        best = [0]
        solved = [False]
        nodes = [0]
        stop = [False]

        def bk(R, P, X):
            if stop[0]:
                return
            if not P and not X:
                if len(R) > best[0]:
                    best[0] = len(R)
                solved[0] = True
                return
            nodes[0] += 1
            if nodes[0] > limit:
                stop[0] = True
                return
            # pivot on max |P & N(u)|
            if P or X:
                u = max(P | X, key=lambda v: len(P & comp[v]))
                cand = P - comp[u]
            else:
                cand = set()
            for v in cand:
                bk(R | {v}, P & comp[v], X & comp[v])
                P.remove(v)
                X.add(v)
                if stop[0]:
                    return

        bk(set(), set(range(m)), set())
        return best[0], solved[0]

    def disjoint_packing_bound(cliques, cover, m, limit=2000000):
        # Exact lower bound = max number of pairwise-disjoint cliques (max IS of
        # the clique-intersection graph). Falls back to a greedy packing if the
        # exact solve is infeasible.
        if m <= 1:
            return m, m > 0
        share = [[False] * m for _ in range(m)]
        cset = [set(c) for c in cliques]
        for i in range(m):
            si = share[i]
            for j in range(i + 1, m):
                if cset[i] & cset[j]:
                    si[j] = share[j][i] = True
        size, solved = max_independent_set_size(m, share, limit)
        if not solved:
            # greedy maximal packing (valid lower bound)
            rem = list(range(m))
            lb = 0
            while rem:
                i = rem.pop(0)
                lb += 1
                rem = [j for j in rem if not share[i][j]]
            size = max(size, lb)
        return size, solved

    def local_search(selected, cliques, cover, n, rng, iters, cset=None):
        selected = set(selected)
        m = len(cliques)
        if cset is None:
            cset = [set(c) for c in cliques]
        for _ in range(iters):
            improved = False
            # 1) removal
            for v in list(selected):
                rest = selected - {v}
                ok = True
                for ci in range(m):
                    if not (rest & cset[ci]):
                        ok = False
                        break
                if ok:
                    selected.discard(v)
                    improved = True
                    break
            if improved:
                continue
            # 2) 1-exchange: remove v, add w
            best = None  # (newsize, w, v)
            for v in selected:
                rest = selected - {v}
                # cliques not hit by rest (must be hit by w)
                need = []
                for ci in range(m):
                    if not (rest & cset[ci]):
                        need.append(ci)
                if not need:
                    continue  # would be a removal (already handled)
                for w in range(n):
                    if w in rest:
                        continue
                    good = True
                    for ci in need:
                        if ci not in cover[w]:
                            good = False
                            break
                    if good:
                        best = (len(selected), w, v)
                        break
                if best:
                    break
            if best:
                _, w, v = best
                selected.discard(v)
                selected.add(w)
                continue
            # 3) 1-swap (same size): remove v add w
            did = False
            for v in list(selected):
                rest = selected - {v}
                for ci in range(m):
                    if not (rest & cset[ci]):
                        cand = [w for w in cset[ci] if w not in rest]
                        if cand:
                            w = rng.choice(cand)
                            selected.discard(v)
                            selected.add(w)
                            did = True
                        break
                if did:
                    break
            if did:
                continue
            # 4) 2-exchange: remove v1, v2, add w1, w2 (same size, can escape 1-exchange minima)
            did2 = False
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    v1, v2 = sel_list[i], sel_list[j]
                    rest = selected - {v1, v2}
                    need = []
                    for ci in range(m):
                        if not (rest & cset[ci]):
                            need.append(ci)
                    if not need:
                        selected.discard(v1)
                        selected.discard(v2)
                        improved = True
                        did2 = True
                        break
                    need_set = set(need)
                    candidates = set()
                    for ci in need:
                        candidates.update(cset[ci])
                    candidates -= rest
                    candidates.discard(v1)
                    candidates.discard(v2)
                    found = False
                    for w1 in candidates:
                        cov1 = cover[w1]
                        remaining_need = need_set - cov1
                        if not remaining_need:
                            continue
                        for w2 in candidates:
                            if w2 == w1:
                                continue
                            if remaining_need <= cover[w2]:
                                selected.discard(v1)
                                selected.discard(v2)
                                selected.add(w1)
                                selected.add(w2)
                                did2 = True
                                found = True
                                break
                        if found:
                            break
                    if did2:
                        break
                if did2:
                    break
            if not did2:
                break
        return selected

    def greedy_cover(cliques, cover, uncovered, rng):
        # uncovered: set of clique indices
        selected = set()
        rem = set(uncovered)
        # counts: for each vertex, number of currently-uncovered cliques it hits
        count = [0] * len(cover)
        for ci in rem:
            for v in cliques[ci]:
                count[v] += 1
        while rem:
            best_v = -1
            best_c = -1
            ties = []
            for v in range(len(cover)):
                cv = count[v]
                if cv <= 0:
                    continue
                if cv > best_c:
                    best_c = cv
                    best_v = v
                    ties = [v]
                elif cv == best_c:
                    ties.append(v)
            if best_v == -1:
                break
            if len(ties) > 1:
                best_v = rng.choice(ties)
            selected.add(best_v)
            for ci in list(cover[best_v]):
                if ci in rem:
                    rem.discard(ci)
                    for v in cliques[ci]:
                        count[v] -= 1
        return selected

    def exact_branch_and_bound(cliques, cover, uncovered, budget_fn, limit=2000000):
        # Returns (best_size, best_set, solved_bool)
        m = len(cliques)
        # Precompute disjointness: disjoint[i][j] = True if cliques i and j share no vertex
        disjoint = [[False] * m for _ in range(m)]
        for i in range(m):
            si = set(cliques[i])
            for j in range(i + 1, m):
                if not si & set(cliques[j]):
                    disjoint[i][j] = disjoint[j][i] = True

        # Precompute clique sizes for ordering
        clique_sizes = [len(c) for c in cliques]

        # Exact lower bound from max disjoint-clique packing (max IS of
        # clique-intersection graph). Tighter than a greedy packing.
        root_lb, root_solved = disjoint_packing_bound(cliques, cover, m)

        # Greedy maximal packing of pairwise-disjoint uncovered cliques (fast,
        # valid lower bound, used inside the recursion)
        def disjoint_lb(uc):
            if not uc:
                return 0
            lb = 0
            rem = set(uc)
            for idx in sorted(rem, key=lambda i: clique_sizes[i]):
                if idx not in rem:
                    continue
                lb += 1
                rem.discard(idx)
                for j in list(rem):
                    if disjoint[idx][j]:
                        rem.discard(j)
            return lb

        best = [len(uncovered) + 1]
        best_set = [set()]
        solved = [False]
        nodes = [0]
        stop = [False]

        def branch(selected, uc):
            if stop[0]:
                return
            if not uc:
                if len(selected) < best[0]:
                    best[0] = len(selected)
                    best_set[0] = set(selected)
                solved[0] = True
                return
            nodes[0] += 1
            if nodes[0] > limit:
                stop[0] = True
                return
            if not budget_fn():
                stop[0] = True
                return
            if len(selected) + disjoint_lb(uc) >= best[0]:
                return
            # choose a minimal uncovered clique
            rep_ci = min(uc, key=lambda i: clique_sizes[i])
            # candidate vertices: those in rep_ci, that are not selected
            verts = [v for v in cliques[rep_ci] if v not in selected]
            # MRV ordering: prefer vertex that covers the fewest uncovered cliques
            def mrv(v):
                c = 0
                for ci in cover[v]:
                    if ci in uc:
                        c += 1
                return c
            verts.sort(key=mrv)
            for v in verts:
                selected.add(v)
                branch(selected, uc - cover[v])
                selected.discard(v)
                if stop[0]:
                    return

        branch(set(), set(uncovered))
        if best[0] <= len(uncovered):
            return best[0], best_set[0], solved[0]
        return None, None, False

    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        if not cliques:
            return []
        cover = make_cover_matrix(cliques, n)
        all_uncovered = frozenset(range(len(cliques)))
        m = len(cliques)
        cset = [set(c) for c in cliques]

        # Exact disjoint-clique packing lower bound (max IS of clique-intersection
        # graph). A transversal must be at least this large.
        lb, _ = disjoint_packing_bound(cliques, cover, m)

        # Greedy + local search baseline
        best = None
        rng = random.Random(777)
        g = greedy_cover(cliques, cover, all_uncovered, rng)
        g = local_search(g, cliques, cover, n, rng, iters=40, cset=cset)
        best = set(g)

        # If the heuristic already matches the packing lower bound, it is optimal.
        if len(best) <= lb:
            return sorted(best)

        # Time budget for exact + extra local search
        def budget_fn():
            return (time.time() - _GLOBAL_START) < _GLOBAL_BUDGET

        # Exact branch-and-bound (with node limit and time guard)
        esize, eset, solved = exact_branch_and_bound(cliques, cover, all_uncovered, budget_fn, limit=5000000)
        if eset and (best is None or len(eset) < len(best)):
            best = set(eset)
        if len(best) <= lb:
            return sorted(best)

        # If exact solved, we're done (optimal)
        if solved:
            return sorted(best)

        # Use remaining time for more greedy restarts + local search
        t0 = time.time()
        for it in range(200):
            if (time.time() - _GLOBAL_START) >= _GLOBAL_BUDGET:
                break
            rr = random.Random(100000 + it)
            cand = greedy_cover(cliques, cover, all_uncovered, rr)
            cand = local_search(cand, cliques, cover, n, rr, iters=30, cset=cset)
            if best is None or len(cand) < len(best):
                best = set(cand)
                if len(best) <= lb:
                    break
            # also try local search starting from current best with fresh rng
            cand2 = local_search(set(best), cliques, cover, n, rr, iters=30, cset=cset)
            if len(cand2) < len(best):
                best = set(cand2)
                if len(best) <= lb:
                    break
        return sorted(best)

    out = []
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    per_instance = (_GLOBAL_BUDGET - (time.time() - _GLOBAL_START)) / len(instances)
    for i, (n, p, seed) in enumerate(instances):
        # hard cap: leave at least a small margin for the rest
        remaining = _GLOBAL_BUDGET - (time.time() - _GLOBAL_START)
        tb = max(1.0, min(per_instance, remaining - 2.0))
        res = solve_instance(n, p, seed, tb)
        out.append(res)
        # if we're running out, just fill the rest cheaply
        if (time.time() - _GLOBAL_START) > _GLOBAL_BUDGET - 1.0:
            for (n2, p2, s2) in instances[i + 1:]:
                out.append(sorted(range(n2)))  # safe (valid) but large
            break
    return out
