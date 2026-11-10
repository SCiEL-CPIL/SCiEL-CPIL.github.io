import time
import random

_GLOBAL_START = time.time()
_GLOBAL_BUDGET = 145.0  # safety margin under the 150 s wall-clock kill

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

    def make_cover_bitmasks(cliques, n):
        # cover_bm[v] = bitmask of cliques that contain v
        m = len(cliques)
        cover_bm = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                cover_bm[v] |= bit
        all_bm = (1 << m) - 1
        return cover_bm, all_bm

    def _popcount(x):
        return x.bit_count()

    def greedy_cover(cliques, cover_bm, all_bm, rng):
        rem = all_bm
        selected = set()
        while rem:
            best_v = -1
            best_c = 0
            ties = []
            for v in range(len(cover_bm)):
                cv = _popcount(cover_bm[v] & rem)
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
            rem &= ~cover_bm[best_v]
        return selected

    def local_search_fast(selected, cliques, cover_bm, all_bm, n, rng, iters):
        selected = set(selected)
        sel_bm = 0
        for v in selected:
            sel_bm |= cover_bm[v]
        for _ in range(iters):
            improved = False
            # 1) removal
            for v in list(selected):
                rest_bm = sel_bm & ~cover_bm[v]
                if rest_bm == all_bm:
                    selected.discard(v)
                    sel_bm = rest_bm
                    improved = True
                    break
            if improved:
                continue
            # 2) 1-exchange: remove v, add w
            best_w = -1
            best_v = -1
            best_gain = 0
            for v in selected:
                rest_bm = sel_bm & ~cover_bm[v]
                need = all_bm & ~rest_bm
                if need == 0:
                    continue
                for w in range(n):
                    if w in selected:
                        continue
                    if (cover_bm[w] & need) == need:
                        new_bm = rest_bm | cover_bm[w]
                        gain = _popcount(new_bm ^ sel_bm)
                        if gain > best_gain:
                            best_gain = gain
                            best_w = w
                            best_v = v
            if best_w >= 0:
                selected.discard(best_v)
                selected.add(best_w)
                sel_bm = (sel_bm & ~cover_bm[best_v]) | cover_bm[best_w]
                continue
            # 3) 2-exchange: remove v1, v2, add w1, w2
            found_2ex = False
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    v1, v2 = sel_list[i], sel_list[j]
                    rest_bm = sel_bm & ~cover_bm[v1] & ~cover_bm[v2]
                    need = all_bm & ~rest_bm
                    if need == 0:
                        continue
                    for w1 in range(n):
                        if w1 in selected:
                            continue
                        bm1 = rest_bm | cover_bm[w1]
                        need1 = all_bm & ~bm1
                        if need1 == 0:
                            # found a valid 2-exchange (size stays same)
                            selected.discard(v1)
                            selected.discard(v2)
                            selected.add(w1)
                            sel_bm = bm1
                            found_2ex = True
                            break
                        for w2 in range(n):
                            if w2 in selected or w2 == w1:
                                continue
                            if (cover_bm[w2] & need1) == need1:
                                new_bm = bm1 | cover_bm[w2]
                                if new_bm == all_bm:
                                    selected.discard(v1)
                                    selected.discard(v2)
                                    selected.add(w1)
                                    selected.add(w2)
                                    sel_bm = new_bm
                                    found_2ex = True
                                    break
                        if found_2ex:
                            break
                if found_2ex:
                    break
            if found_2ex:
                continue
            # 4) 1-swap (same size): remove v add w, try to improve coverage
            did = False
            for v in list(selected):
                rest_bm = sel_bm & ~cover_bm[v]
                for w in range(n):
                    if w in selected:
                        continue
                    new_bm = rest_bm | cover_bm[w]
                    if new_bm == all_bm:
                        # valid swap, check if it helps future removals (heuristic: prefer w with more total coverage)
                        selected.discard(v)
                        selected.add(w)
                        sel_bm = new_bm
                        did = True
                        break
                if did:
                    break
            if not did:
                break
        return selected

    def exact_branch_and_bound(cliques, cover_bm, all_bm, budget_fn, limit=3000000):
        n = len(cover_bm)
        m = len(cliques)
        best_size = [10**9]
        best_set = [set()]
        solved = [False]
        nodes = [0]
        stop = [False]

        # Precompute for each clique the list of vertices in it
        def branch(selected, sel_bm):
            if stop[0]:
                return
            if sel_bm == all_bm:
                if len(selected) < best_size[0]:
                    best_size[0] = len(selected)
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

            # Lower bound: greedy packing of disjoint uncovered cliques
            unhit = all_bm & ~sel_bm
            lb = 0
            rem = unhit
            while rem:
                # Pick the unhit clique with the fewest vertices (most constrained)
                best_ci = -1
                best_size_ci = 10**9
                for ci in range(m):
                    if (rem >> ci) & 1:
                        sz = len(cliques[ci])
                        if sz < best_size_ci:
                            best_size_ci = sz
                            best_ci = ci
                if best_ci == -1:
                    break
                lb += 1
                # Remove all cliques that intersect with clique best_ci
                rep = cliques[best_ci]
                new_rem = rem
                for v in rep:
                    new_rem &= ~cover_bm[v]
                rem = new_rem
                if not rem:
                    break
            # Additional lower bound: count of uncovered cliques / max clique size
            # Actually a simpler bound: each vertex can cover at most max_clique_size cliques
            # But the packing bound is usually tighter. Let's keep it.

            if len(selected) + lb >= best_size[0]:
                return

            # Choose a minimal uncovered clique (by size)
            unhit_ci_list = []
            for ci in range(m):
                if (unhit >> ci) & 1:
                    unhit_ci_list.append(ci)
            if not unhit_ci_list:
                solved[0] = True
                return
            rep_ci = min(unhit_ci_list, key=lambda i: len(cliques[i]))

            # Candidate vertices: those in rep_ci, ordered by descending coverage of unhit cliques
            verts = list(cliques[rep_ci])
            verts.sort(key=lambda v: _popcount(cover_bm[v] & unhit), reverse=True)

            for v in verts:
                if v in selected:
                    continue
                new_bm = sel_bm | cover_bm[v]
                selected.add(v)
                branch(selected, new_bm)
                selected.discard(v)
                if stop[0]:
                    return
                if len(selected) + 1 >= best_size[0]:
                    # Even if we add no more, we can't beat best
                    pass

        branch(set(), 0)
        if best_size[0] < 10**9:
            return best_size[0], best_set[0], solved[0]
        return None, None, False

    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        if not cliques:
            return []
        cover_bm, all_bm = make_cover_bitmasks(cliques, n)

        # Greedy + local search baseline
        best = None
        rng = random.Random(777)
        g = greedy_cover(cliques, cover_bm, all_bm, rng)
        g = local_search_fast(g, cliques, cover_bm, all_bm, n, rng, iters=50)
        best = set(g)

        # Time budget for exact + extra local search
        def budget_fn():
            return (time.time() - _GLOBAL_START) < _GLOBAL_BUDGET

        # Exact branch-and-bound (with node limit and time guard)
        esize, eset, solved = exact_branch_and_bound(cliques, cover_bm, all_bm, budget_fn)
        if eset and (best is None or len(eset) < len(best)):
            best = set(eset)

        # If exact solved, we're done (optimal)
        if solved:
            return sorted(best)

        # Use remaining time for more greedy restarts + local search
        t0 = time.time()
        for it in range(500):
            if (time.time() - _GLOBAL_START) >= _GLOBAL_BUDGET:
                break
            rr = random.Random(100000 + it)
            cand = greedy_cover(cliques, cover_bm, all_bm, rr)
            cand = local_search_fast(cand, cliques, cover_bm, all_bm, n, rr, iters=60)
            if best is None or len(cand) < len(best):
                best = set(cand)
            # also try local search starting from current best with fresh rng
            cand2 = local_search_fast(set(best), cliques, cover_bm, all_bm, n, rr, iters=60)
            if len(cand2) < len(best):
                best = set(cand2)
        return sorted(best)

    out = []
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    # Allocate time proportional to problem size (n^2 heuristic)
    weights = [n * n for n, p, s in instances]
    total_w = sum(weights)
    remaining_total = _GLOBAL_BUDGET - (time.time() - _GLOBAL_START)
    time_alloc = [w / total_w * remaining_total for w in weights]
    
    for i, (n, p, seed) in enumerate(instances):
        # hard cap: leave at least a small margin for the rest
        remaining = _GLOBAL_BUDGET - (time.time() - _GLOBAL_START)
        tb = max(1.0, min(time_alloc[i], remaining - 2.0))
        res = solve_instance(n, p, seed, tb)
        out.append(res)
        # if we're running out, just fill the rest cheaply
        if (time.time() - _GLOBAL_START) > _GLOBAL_BUDGET - 1.0:
            for (n2, p2, s2) in instances[i + 1:]:
                out.append(sorted(range(n2)))  # safe (valid) but large
            break
    return out
