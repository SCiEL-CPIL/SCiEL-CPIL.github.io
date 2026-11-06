import math
import time
import random

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before the 90s wall-clock kill
    random.seed(42)

    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2*a must never be a square.
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)

    # Conflicts restricted to the candidate set.
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    # Greedy coloring upper bound for independent set on a subgraph.
    # Returns the number of colors used, which upper-bounds the MIS size.
    def color_bound(nodes):
        if not nodes:
            return 0
        # Order by degree descending for better coloring
        degs = [(len(conflict[u] & nodes), u) for u in nodes]
        degs.sort(reverse=True)
        ordered = [u for _, u in degs]
        
        color_of = {}
        next_color = 0
        for u in ordered:
            # Find smallest available color
            used = set()
            for v in conflict[u]:
                if v in nodes and v in color_of:
                    used.add(color_of[v])
            c = 0
            while c in used:
                c += 1
            color_of[u] = c
            if c > next_color:
                next_color = c
        return next_color + 1

    # Independent-set maximizer over the candidate pool.
    # Uses a greedy lower bound + exact branch-and-bound with coloring upper bound.
    def maximize(pool, time_budget):
        pool = set(pool)
        # greedy independent set (min-degree ordering) for a solid lower bound
        rem = set(pool)
        greedy = []
        while rem:
            best = None
            bd = None
            for node in rem:
                d = len(conflict[node] & rem)
                if bd is None or d < bd:
                    bd, best = d, node
                    if d == 0:
                        break
            greedy.append(best)
            rem.discard(best)
            rem -= conflict[best]
        best = set(greedy)
        blen = len(best)

        def time_up():
            return time.time() - start > time_budget

        # branch-and-bound on a node subset `sub` with fixed `fixed`
        def search(sub, fixed):
            nonlocal best, blen
            if not sub:
                if len(fixed) > blen:
                    blen, best = len(fixed), set(fixed)
                return
            if time_up():
                return
            # Use coloring bound as upper bound (much tighter than components)
            ub = len(fixed) + color_bound(sub)
            if ub <= blen:
                return
            # pick a high-degree vertex in `sub` to branch on
            v = None
            hv = -1
            for u in sub:
                d = len(conflict[u] & sub)
                if d > hv:
                    hv, v = d, u
            # branch 1: include v
            sub_include = (sub - {v}) - conflict[v]
            if len(fixed) + 1 + color_bound(sub_include) > blen:
                search(sub_include, fixed | {v})
            # branch 2: exclude v
            sub_exclude = sub - {v}
            if len(fixed) + color_bound(sub_exclude) > blen:
                search(sub_exclude, fixed)

        # initial search over the whole pool
        search(pool, set())

        # component decomposition: solve each connected component exactly,
        # union the solutions (cross-component vertices never conflict)
        comps = []
        seen = set()
        for start_node in pool:
            if start_node in seen:
                continue
            comp = set()
            stack = [start_node]
            seen.add(start_node)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in conflict[u]:
                    if v in pool and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
            if time_up():
                break
        union = set()
        for c in comps:
            cbl = [0]
            cbest = [set()]
            def s2(sub, fixed):
                if not sub:
                    if len(fixed) > cbl[0]:
                        cbl[0] = len(fixed)
                        cbest[0] = set(fixed)
                    return
                if time.time() - start > time_budget:
                    return
                if len(fixed) + color_bound(sub) <= cbl[0]:
                    return
                v = None
                hv = -1
                for u in sub:
                    d = len(conflict[u] & sub)
                    if d > hv:
                        hv, v = d, u
                sub_inc = (sub - {v}) - conflict[v]
                if len(fixed) + 1 + color_bound(sub_inc) > cbl[0]:
                    s2(sub_inc, fixed | {v})
                sub_exc = sub - {v}
                if len(fixed) + color_bound(sub_exc) > cbl[0]:
                    s2(sub_exc, fixed)
            s2(c, set())
            union |= cbest[0]
            if time_up():
                break
        if len(union) > blen:
            best, blen = set(union), len(union)
        return best

    def local_search(solution, time_budget):
        """
        Swap-based local search with 'remove 1, add k' moves.
        Uses set intersection for O(1) conflict checks.
        """
        sol = set(solution)
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            cur_size = len(sol)
            
            # Phase 1: Try removing one element and greedily adding as many as possible
            for a in list(sol):
                if time.time() - start > time_budget:
                    return sol
                trial = sol - {a}
                # Find all addable candidates
                addable = []
                for b in candidates:
                    if b in trial:
                        continue
                    if conflict[b] & trial:
                        continue
                    addable.append(b)
                
                if not addable:
                    continue
                    
                # Greedily add from addable using min-degree heuristic
                added = []
                rem_addable = set(addable)
                while rem_addable:
                    # Pick node with min degree in remaining addable
                    best_node = None
                    bd = None
                    for node in rem_addable:
                        d = len(conflict[node] & rem_addable)
                        if bd is None or d < bd:
                            bd, best_node = d, node
                            if d == 0:
                                break
                    added.append(best_node)
                    rem_addable.discard(best_node)
                    rem_addable -= conflict[best_node]
                
                new_size = len(trial) + len(added)
                if new_size > cur_size:
                    sol = trial | set(added)
                    cur_size = new_size
                    improved = True
                    break
            
            # Phase 2: Single swaps (remove a, add b) if no improvement yet
            if not improved:
                for a in list(sol):
                    if time.time() - start > time_budget:
                        return sol
                    trial = sol - {a}
                    for b in candidates:
                        if b in trial or b == a:
                            continue
                        if conflict[b] & trial:
                            continue
                        # Try adding b, then greedily fill
                        new_sol = trial | {b}
                        fill = set()
                        for c in candidates:
                            if c in new_sol:
                                continue
                            if not (conflict[c] & (new_sol | fill)):
                                fill.add(c)
                        if len(new_sol) + len(fill) > cur_size:
                            sol = new_sol | fill
                            cur_size = len(sol)
                            improved = True
                            break
                    if improved:
                        break
        return sol

    def randomized_greedy(pool, time_budget):
        """Randomized greedy with min-degree bias."""
        rem = set(pool)
        result = []
        while rem and time.time() - start < time_budget:
            scored = []
            for node in rem:
                d = len(conflict[node] & rem)
                score = d + random.random() * 0.5
                scored.append((score, node))
            scored.sort()
            best = scored[0][1]
            result.append(best)
            rem.discard(best)
            rem -= conflict[best]
        return set(result)

    best_solution = set()

    # 1. Exact/branch-and-bound on the full candidate pool
    sol = maximize(candidates, deadline - 15.0)
    best_solution = set(sol)

    # 2. Local augmentation: add any candidate that conflicts with nothing in the set.
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            best_solution.add(a)

    # 3. Local search with swap improvements
    best_solution = local_search(best_solution, deadline - 5.0)

    # 4. Randomized restarts with local search
    if time.time() - start < deadline - 3.0:
        rng = random.Random(42)
        num_restarts = 20
        for _ in range(num_restarts):
            if time.time() - start > deadline - 3.0:
                break
            r_sol = randomized_greedy(candidates, deadline - 3.0)
            r_sol = local_search(r_sol, deadline - 3.0)
            if len(r_sol) > len(best_solution):
                best_solution = r_sol

    # 5. Safety verification: ensure no pair (a<=b) has a square sum.
    ok = True
    lst = sorted(best_solution)
    for i in range(len(lst)):
        a = lst[i]
        if 2 * a in squares:
            ok = False
            break
        for j in range(i + 1, len(lst)):
            if a + lst[j] in squares:
                ok = False
                break
        if not ok:
            break
    if not ok:
        # fallback: deterministic min-degree greedy (always valid)
        rem = set(candidates)
        fb = []
        while rem:
            mn = None
            md = None
            for node in rem:
                d = len(conflict[node] & rem)
                if md is None or d < md:
                    md, mn = d, node
            fb.append(mn)
            rem.discard(mn)
            rem -= conflict[mn]
        best_solution = set(fb)

    return sorted(best_solution)
