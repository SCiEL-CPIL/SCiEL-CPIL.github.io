<<<<<<< SEARCH
    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        if not cliques:
            return []
        cover = make_cover_matrix(cliques, n)
        all_uncovered = frozenset(range(len(cliques)))

        # Greedy + local search baseline
        best = None
        rng = random.Random(777)
        g = greedy_cover(cliques, cover, all_uncovered, rng)
        g = local_search(g, cliques, cover, n, rng, iters=40)
        best = set(g)

        # Time budget for exact + extra local search
        def budget_fn():
            return (time.time() - _GLOBAL_START) < _GLOBAL_BUDGET

        # Exact branch-and-bound (with node limit and time guard)
        esize, eset, solved = exact_branch_and_bound(cliques, cover, all_uncovered, budget_fn)
        if eset and (best is None or len(eset) < len(best)):
            best = set(eset)

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
            cand = local_search(cand, cliques, cover, n, rr, iters=30)
            if best is None or len(cand) < len(best):
                best = set(cand)
            # also try local search starting from current best with fresh rng
            cand2 = local_search(set(best), cliques, cover, n, rr, iters=30)
            if len(cand2) < len(best):
                best = set(cand2)
        return sorted(best)
=======
    def solve_instance(n, p, seed, time_budget):
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        if not cliques:
            return []
        cover = make_cover_matrix(cliques, n)
        all_uncovered = frozenset(range(len(cliques)))

        # Build bitmask coverage
        num_cliques = len(cliques)
        cover_bm = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                cover_bm[v] |= bit
        all_bm = (1 << num_cliques) - 1

        # Greedy + fast local search baseline
        rng = random.Random(777)
        g = greedy_cover(cliques, cover, all_uncovered, rng)
        g = local_search_fast(g, cliques, cover_bm, all_bm, n, rng, iters=50)
        best = set(g)

        # Time budget for exact + extra local search
        def budget_fn():
            return (time.time() - _GLOBAL_START) < _GLOBAL_BUDGET

        # Exact branch-and-bound (with node limit and time guard)
        eset, solved = exact_branch_and_bound(cliques, cover_bm, all_bm, budget_fn)
        if eset and len(eset) < len(best):
            best = set(eset)

        # If exact solved, we're done (optimal)
        if solved:
            return sorted(best)

        # Use remaining time for more greedy restarts + fast local search
        for it in range(500):
            if (time.time() - _GLOBAL_START) >= _GLOBAL_BUDGET:
                break
            rr = random.Random(100000 + it)
            cand = greedy_cover(cliques, cover, all_uncovered, rr)
            cand = local_search_fast(cand, cliques, cover_bm, all_bm, n, rr, iters=50)
            if len(cand) < len(best):
                best = set(cand)
            # also try local search starting from current best with fresh rng
            cand2 = local_search_fast(set(best), cliques, cover_bm, all_bm, n, rr, iters=30)
            if len(cand2) < len(best):
                best = set(cand2)
        return sorted(best)
>>>>>>> REPLACE
