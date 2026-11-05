import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def enumerate_maximal_cliques(adj, n):
        cliques = []
        full = (1 << n) - 1

        def bk(R, P, X):
            if P == 0 and X == 0:
                cliques.append(R)
                return
            # pivot: maximize |P ∩ N(u)| over u in P∪X
            PX = P | X
            if PX:
                best = -1
                pivot = -1
                t = PX
                while t:
                    lsb = t & -t
                    u = lsb.bit_length() - 1
                    c = (P & adj[u]).bit_count()
                    if c > best:
                        best = c
                        pivot = u
                    t ^= lsb
                candidates = P & ~adj[pivot]
            else:
                candidates = P
            t = candidates
            while t:
                lsb = t & -t
                v = lsb.bit_length() - 1
                t ^= lsb
                bk(R | lsb, P & adj[v], X & adj[v])
                P ^= lsb
                X |= lsb

        bk(0, full, 0)
        return cliques

    def solve_instance(adj, n, time_budget):
        cliques = enumerate_maximal_cliques(adj, n)
        # filter size >= 2
        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return []

        # Order cliques: smaller first (harder to satisfy), then by index for determinism
        cliques.sort(key=lambda c: (c.bit_count(), c))
        m = len(cliques)
        # uncovered: bit i set iff clique i is not yet hit
        uncovered = (1 << m) - 1
        if uncovered == 0:
            return []

        selected = []

        # Greedy set cover
        while uncovered and time.time() < deadline - 5.0:
            best_v = -1
            best_hits = 0
            for v in range(n):
                mask = 1 << v
                hits = 0
                u = uncovered
                while u:
                    lsb = u & -u
                    idx = lsb.bit_length() - 1
                    if cliques[idx] & mask:
                        hits += 1
                    u ^= lsb
                if hits > best_hits:
                    best_hits = hits
                    best_v = v
            if best_v == -1 or best_hits == 0:
                break
            selected.append(best_v)
            vmask = 1 << best_v
            u = uncovered
            new_unc = 0
            while u:
                lsb = u & -u
                idx = lsb.bit_length() - 1
                if not (cliques[idx] & vmask):
                    new_unc |= lsb
                u ^= lsb
            uncovered = new_unc

        # Fallback: ensure completeness
        if uncovered:
            u = uncovered
            while u:
                lsb = u & -u
                idx = lsb.bit_length() - 1
                c = cliques[idx]
                lsb_c = c & -c
                v = lsb_c.bit_length() - 1
                if v not in selected:
                    selected.append(v)
                u ^= lsb
            uncovered = 0

        # ---- Local search polish ----
        sel_set = set(selected)
        # Precompute for each selected vertex the set of cliques it uniquely/also hits
        # We'll recompute validity by scanning.

        def is_valid(sel):
            s = set(sel)
            for c in cliques:
                if c & 0 == 0:
                    continue
                # check intersection
                t = c
                ok = False
                while t:
                    lsb = t & -t
                    vv = lsb.bit_length() - 1
                    if vv in s:
                        ok = True
                        break
                    t ^= lsb
                if not ok:
                    return False
            return True

        # Removal pass: repeatedly try to remove each vertex
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # iterate in a fixed order; try removal
            for i in range(len(selected) - 1, -1, -1):
                v = selected[i]
                new_sel = selected[:i] + selected[i+1:]
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break  # restart scan after a change

        # Swap pass: 1-for-1 (remove one selected, add one unselected) to reduce? 
        # Actually swap keeps size same; we want size reduction. 
        # Try 2-for-1: remove two selected, add one unselected, if valid and reduces by 1.
        # Simpler: try removing pairs and see if a single vertex covers the gap.
        # We'll do: for each pair (a,b) in selected, try replacing with a single unselected v.
        # This reduces size by 1 if valid.
        if time.time() < deadline - 2.0:
            n_sel = len(selected)
            progress = True
            while progress and time.time() < deadline - 2.0:
                progress = False
                n_sel = len(selected)
                for i in range(n_sel):
                    for j in range(i + 1, n_sel):
                        a = selected[i]
                        b = selected[j]
                        # candidate replacement: any unselected vertex
                        for v in range(n):
                            if v == a or v == b or v in sel_set:
                                continue
                            new_sel = [x for x in selected if x != a and x != b] + [v]
                            if is_valid(new_sel):
                                selected = new_sel
                                sel_set = set(selected)
                                progress = True
                                break
                        if progress:
                            break
                    if progress:
                        break

        # Final removal pass after swaps
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(len(selected) - 1, -1, -1):
                v = selected[i]
                new_sel = selected[:i] + selected[i+1:]
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break

        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            adj = random_graph(n, p, seed)
            results.append(list(range(n)))
            continue
        adj = random_graph(n, p, seed)
        transversal = solve_instance(adj, n, deadline - time.time())
        results.append(transversal)

    return results
