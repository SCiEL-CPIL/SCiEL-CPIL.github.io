def construct():
    import time
    from math import gcd
    from itertools import combinations
    import random

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
    pairs = [set() for _ in range(N + 1)]
    for (a, b, c) in bad:
        pairs[a].add((b, c))
        pairs[b].add((a, c))
        pairs[c].add((a, b))

    # Precompute pair lookup as a set of frozensets for faster membership
    pairs_fs = [set() for _ in range(N + 1)]
    for i in range(N + 1):
        for p in pairs[i]:
            pairs_fs[i].add(frozenset(p))

    # ---------- Fast feasibility test ----------
    def can_add(x, S_set):
        """S is a set of chosen vertices; can x join without creating a bad triple?"""
        ps = pairs_fs[x]
        # need u,v in S with {u,v} in ps
        for u in S_set:
            for v in S_set:
                if u == v:
                    continue
                if frozenset((u, v)) in ps:
                    return False
        return True

    # Even faster: precompute for each x, which pairs conflict
    # Use a different approach: for each x, build a set of all vertices that form a bad pair with some other
    # Actually, let's use a bitset approach for speed
    # Represent each vertex's conflict pairs as a set of (min, max) tuples
    # For can_add, we need to check if any pair in S is in conflicts[x]
    
    # Better: for each x, create a set of all y such that there exists z with {x,y,z} bad
    # No, that's not right. We need: for fixed x and set S, is there u,v in S with {x,u,v} bad?
    # This is equivalent to: does S contain any edge from the graph G_x where edges are {u,v} | {x,u,v} bad?
    
    # Let's build for each x, the graph G_x as a set of edges (u,v) with u<v
    graph_edges = [set() for _ in range(N + 1)]
    for (a, b, c) in bad:
        for x in (a, b, c):
            others = [y for y in (a, b, c) if y != x]
            u, v = min(others), max(others)
            graph_edges[x].add((u, v))

    def can_add_fast(x, S_list, S_set):
        """Check if x can be added to S without creating a bad triple."""
        edges_x = graph_edges[x]
        if not edges_x:
            return True
        # Check all pairs in S
        n = len(S_list)
        for i in range(n):
            u = S_list[i]
            for j in range(i + 1, n):
                v = S_list[j]
                if (u, v) in edges_x or (v, u) in edges_x:
                    return False
        return True

    all_numbers = list(range(1, N + 1))

    # ---------- Helper functions ----------
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

    # ---------- Greedy with multiple strategies ----------
    def greedy(order):
        selected = []
        sel_set = set()
        for x in order:
            if x in sel_set:
                continue
            if can_add_fast(x, selected, sel_set):
                selected.append(x)
                sel_set.add(x)
        return selected

    def greedy_random():
        """Randomized greedy: shuffle and greedily add."""
        order = all_numbers[:]
        random.shuffle(order)
        return greedy(order)

    # Deterministic orderings
    orderings = [
        sorted(all_numbers, key=lambda x: (degree[x], x)),
        sorted(all_numbers, key=lambda x: (-num_distinct_pf(x), -total_pf(x), -num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (-total_pf(x), -num_distinct_pf(x), x)),
        sorted(all_numbers, key=lambda x: (-num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (-degree[x], x)),
        sorted(all_numbers, key=lambda x: (x,)),  # natural order
        sorted(all_numbers, key=lambda x: (-x,)),  # reverse natural
    ]

    best = []
    best_set = set()

    # Try deterministic orderings
    for ord_ in orderings:
        r = greedy(ord_)
        if len(r) > len(best):
            best = r
            best_set = set(r)

    # Try random restarts
    num_restarts = 20
    for _ in range(num_restarts):
        if time_left() < 5:
            break
        r = greedy_random()
        if len(r) > len(best):
            best = r
            best_set = set(r)

    # ---------- Local search: add, swap, remove+add2 ----------
    def local_search(selected, max_time=10):
        """Improve selected via local moves."""
        selected = list(selected)
        sel_set = set(selected)
        unselected = [x for x in all_numbers if x not in sel_set]
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Try single additions
            for x in unselected:
                if can_add_fast(x, selected, sel_set):
                    selected.append(x)
                    sel_set.add(x)
                    unselected = [u for u in all_numbers if u not in sel_set]
                    improved = True
            
            # Try 1-for-1 swaps that enable addition
            for x in unselected:
                if time_left() < 5:
                    break
                for y in list(selected):
                    cand = [s for s in selected if s != y]
                    cand_set = set(cand)
                    if can_add_fast(x, cand, cand_set):
                        cand.append(x)
                        cand_set.add(x)
                        # Try to add another
                        for z in unselected:
                            if z == x:
                                continue
                            if can_add_fast(z, cand, cand_set):
                                selected = cand
                                sel_set = cand_set
                                unselected = [u for u in all_numbers if u not in sel_set]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
            
            # Try remove one, add two
            if time_left() > 10:
                for y in list(selected):
                    if time_left() < 10:
                        break
                    temp = [s for s in selected if s != y]
                    temp_set = set(temp)
                    added = []
                    for x1 in unselected:
                        if can_add_fast(x1, temp, temp_set):
                            temp.append(x1)
                            temp_set.add(x1)
                            added.append(x1)
                            for x2 in unselected:
                                if x2 in added:
                                    continue
                                if can_add_fast(x2, temp, temp_set):
                                    temp.append(x2)
                                    temp_set.add(x2)
                                    added.append(x2)
                                    break
                            if len(added) == 2:
                                selected = temp
                                sel_set = temp_set
                                unselected = [u for u in all_numbers if u not in sel_set]
                                improved = True
                                break
                    if improved:
                        break
        
        return selected

    # Apply local search to best
    best = local_search(best, max_time=15)
    best_set = set(best)

    # Try local search from multiple starting points
    if time_left() > 20:
        for _ in range(10):
            if time_left() < 10:
                break
            r = greedy_random()
            r = local_search(r, max_time=5)
            if len(r) > len(best):
                best = r
                best_set = set(r)

    # ---------- Final polish: exhaustive add check ----------
    if time_left() > 5:
        sel_set = set(best)
        unselected = [x for x in all_numbers if x not in sel_set]
        changed = True
        while changed and time_left() > 5:
            changed = False
            for x in unselected:
                if can_add_fast(x, best, sel_set):
                    best.append(x)
                    sel_set.add(x)
                    unselected = [u for u in all_numbers if u not in sel_set]
                    changed = True

    if not best:
        best = [1]
    return best
