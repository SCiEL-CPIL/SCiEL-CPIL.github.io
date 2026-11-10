def construct():
    import time
    from math import gcd
    from itertools import combinations

    deadline = time.time() + 82  # safety margin under 90s kill

    def time_left():
        return deadline - time.time()

    # ---------- Precompute gcd table ----------
    N = 128
    gcd_t = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            gcd_t[a][b] = g
            gcd_t[b][a] = g

    # ---------- Build 3-uniform hypergraph of "bad" triples ----------
    # A triple {a,b,c} is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_t[a][b]
            for c in range(b + 1, N + 1):
                if gcd_t[a][c] == gab and gcd_t[b][c] == gab:
                    bad.add((a, b, c))

    # For each vertex x, the set of (u,v) pairs such that {x,u,v} is bad.
    # x is incompatible with a set S if there exist u,v in S with (u,v) in pairs[x].
    pairs = [set() for _ in range(N + 1)]
    for (a, b, c) in bad:
        pairs[a].add((b, c))
        pairs[b].add((a, c))
        pairs[c].add((a, b))

    # ---------- Fast feasibility test ----------
    def can_add(x, S):
        """S is a frozenset/list of chosen vertices; can x join without creating a bad triple?"""
        ps = pairs[x]
        # need u,v in S with (u,v) in ps (in any order)
        for u in S:
            for v in S:
                if u == v:
                    continue
                if (u, v) in ps:
                    return False
        return True

    def can_add_fast(x, S_list, S_set):
        ps = pairs[x]
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i + 1, len(S_list)):
                v = S_list[j]
                if (u, v) in ps or (v, u) in ps:
                    return False
        return True

    all_numbers = list(range(1, N + 1))

    # ---------- Greedy seeding (multiple orderings) ----------
    def num_distinct_pf(n):
        cnt = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt

    def total_pf(n):
        cnt = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                cnt += 1
                n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt

    def num_divisors(n):
        cnt = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                cnt += 1
                if i != n // i:
                    cnt += 1
            i += 1
        return cnt

    # hyperedge degree: number of bad triples containing x
    degree = [0] * (N + 1)
    for (a, b, c) in bad:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1

    def greedy(order):
        selected = []
        sel_set = set()
        for x in order:
            if x in sel_set:
                continue
            if can_add(x, sel_set):
                selected.append(x)
                sel_set.add(x)
        return selected

    best = []
    best_set = set()

    orderings = [
        sorted(all_numbers, key=lambda x: (degree[x], x)),          # low degree first
        sorted(all_numbers, key=lambda x: (-num_distinct_pf(x), -total_pf(x), -num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (-total_pf(x), -num_distinct_pf(x), x)),
        sorted(all_numbers, key=lambda x: (-num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (-degree[x], x)),
    ]
    for ord_ in orderings:
        r = greedy(ord_)
        if len(r) > len(best):
            best = r
            best_set = set(r)

    # ---------- Branch & bound maximum independent set ----------
    # Vertices = all_numbers. We want max subset with no bad triple.
    # Use recursive search with upper bound = |current| + |remaining| (trivial)
    # plus a tighter bound via greedy on remaining.

    best_size = len(best)
    best_found = list(best)

    # Precompute a greedy bound function on a candidate pool
    def greedy_size(pool):
        s = 0
        for x in pool:
            # quick greedy: just count (approx) — use actual greedy on pool
            pass
        # do real greedy
        sel = []
        selset = set()
        for x in pool:
            if can_add(x, selset):
                sel.append(x)
                selset.add(x)
        return len(sel)

    # Upper bound: current_size + greedy_size(remaining)
    def upper_bound(cur_size, remaining):
        if not remaining:
            return cur_size
        return cur_size + greedy_size(remaining)

    # Branch and bound
    # Order vertices by degree to prune: high degree vertices are more constraining.
    # We search over inclusion/exclusion.

    def bb(selected_list, selected_set, remaining, depth):
        nonlocal best_size, best_found
        if time_left() < 1.0:
            return
        cur = len(selected_list)
        if cur + len(remaining) <= best_size:
            return  # can't beat best
        if not remaining:
            if cur > best_size:
                best_size = cur
                best_found = list(selected_list)
            return
        # tighter bound
        ub = cur + greedy_size(remaining)
        if ub <= best_size:
            return

        # Pick next vertex: highest degree among remaining (most constraining)
        x = max(remaining, key=lambda v: degree[v])
        rem_list = list(remaining)
        rem_list.remove(x)

        # Branch 1: include x
        if can_add(x, selected_set):
            new_sel = selected_list + [x]
            new_set = selected_set | {x}
            # remaining must be a subset compatible... but we keep all remaining and rely on can_add later
            bb(new_sel, new_set, set(rem_list), depth + 1)

        # Branch 2: exclude x
        bb(selected_list, selected_set, set(rem_list), depth + 1)

    # Start B&B from the greedy best as initial lower bound.
    # To keep it tractable, run B&B but with a global time guard.
    # We search over all numbers; remaining = all numbers.
    # But that's 2^128. We rely on the bound to prune.
    # Start with empty selection; the bound greedy_size(all) gives the greedy optimum.
    # Since best_size already = greedy optimum, B&B will only explore branches
    # that could exceed it.

    # To make B&B effective, seed remaining as all numbers but we already have a
    # strong lower bound, so most branches prune immediately.
    # Use a smarter start: pick a vertex ordering and go.
    full = set(all_numbers)
    # We do NOT include the greedy best as forced; we search globally.
    # But to be safe and fast, we cap depth and rely on time.
    bb([], set(), full, 0)

    # ---------- Local search improvement on best_found ----------
    selected = best_found if len(best_found) > len(best) else best
    best = selected
    best_set = set(selected)
    unselected = [x for x in all_numbers if x not in best_set]

    improved = True
    while improved and time_left() > 5:
        improved = False
        # Try single additions
        for x in unselected:
            if can_add(x, best_set):
                best.append(x)
                best_set.add(x)
                unselected = [u for u in all_numbers if u not in best_set]
                improved = True
        # Try 1-for-1 swaps
        for x in unselected:
            if time_left() < 5:
                break
            for y in list(best):
                cand = [s for s in best if s != y]
                cand_set = set(cand)
                if can_add(x, cand_set):
                    # accept swap only if it allows a later addition; here same size
                    # Do a 2-step: swap then try to add another
                    cand.append(x)
                    cand_set.add(x)
                    for z in unselected:
                        if z == x:
                            continue
                        if can_add(z, cand_set):
                            # net +1
                            best = cand
                            best_set = cand_set
                            unselected = [u for u in all_numbers if u not in best_set]
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break

    # ---------- Final local search: try removing one and adding two ----------
    if time_left() > 8:
        improved = True
        while improved and time_left() > 8:
            improved = False
            for y in list(best):
                if time_left() < 8:
                    break
                temp = [s for s in best if s != y]
                temp_set = set(temp)
                added = []
                for x1 in unselected:
                    if can_add(x1, temp_set):
                        temp.append(x1)
                        temp_set.add(x1)
                        added.append(x1)
                        for x2 in unselected:
                            if x2 in added:
                                continue
                            if can_add(x2, temp_set):
                                temp.append(x2)
                                temp_set.add(x2)
                                added.append(x2)
                                break
                        if len(added) == 2:
                            best = temp
                            best_set = temp_set
                            unselected = [u for u in all_numbers if u not in best_set]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    # Final safety: ensure nonempty
    if not best:
        best = [1]
    return best
