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

    # Fast upper bound: every connected component of the induced conflict graph
    # contributes at most 1 vertex to any independent set.
    comp_id = {}
    def _components(nodes):
        seen = set()
        comps = 0
        for start_node in nodes:
            if start_node in seen:
                continue
            comps += 1
            stack = [start_node]
            seen.add(start_node)
            while stack:
                u = stack.pop()
                for v in conflict[u]:
                    if v in nodes and v not in seen:
                        seen.add(v)
                        stack.append(v)
        return comps

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
            sub_best = set()
            blen_c = 0
            # local exact solve on this component
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

    best_solution = set()

    # 1. Exact/branch-and-bound on the full candidate pool (fast due to matching structure).
    sol = maximize(candidates, deadline - 3.0)
    best_solution = set(sol)

    # 2. Local augmentation: add any candidate that conflicts with nothing in the set.
    improved = list(best_solution)
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            improved.append(a)
            best_solution.add(a)

    # 3. 1-swap local search: try replacing one element to free up space for more
    def local_search_swap(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and (time.time() - start) < time_budget:
            improved_flag = False
            cur_size = len(s)
            # Try removing each element and then adding any non-conflicting candidate
            for rem in list(s):
                s2 = s - {rem}
                # Find all candidates that can be added to s2
                addable = []
                for a in candidates:
                    if a not in s2 and not (conflict[a] & s2):
                        addable.append(a)
                if len(addable) >= 1:
                    # Best case: we can add at least 1, net change is +1-1=0 or better
                    # Actually we removed 1, so if we can add k, net is k-1
                    # We want k >= 2 for improvement, or k >= 1 to maintain size
                    # For improvement: need len(addable) >= 2 (remove 1, add 2)
                    # But actually we just need to find if any single swap helps
                    # Try adding each addable one by one
                    for a in addable:
                        s3 = s2 | {a}
                        # Check if we can add more after this
                        # Just do a greedy fill after the swap
                        s4 = set(s3)
                        # Greedy add more
                        changed = True
                        while changed and (time.time() - start) < time_budget:
                            changed = False
                            for b in candidates:
                                if b not in s4 and not (conflict[b] & s4):
                                    s4.add(b)
                                    changed = True
                        if len(s4) > cur_size:
                            s = s4
                            cur_size = len(s)
                            improved_flag = True
                            break
                if improved_flag:
                    break
        return s

    best_solution = local_search_swap(best_solution, deadline - 3.0)

    # 4. Multiple random restarts of greedy + local search if time permits
    if (time.time() - start) < deadline - 5.0:
        for restart in range(20):
            if (time.time() - start) >= deadline - 3.0:
                break
            # Randomized greedy
            rem = set(candidates)
            greedy_set = set()
            while rem:
                # Pick random low-degree node
                low_deg_nodes = []
                min_d = float('inf')
                for node in rem:
                    d = len(conflict[node] & rem)
                    if d < min_d:
                        min_d = d
                        low_deg_nodes = [node]
                    elif d == min_d:
                        low_deg_nodes.append(node)
                chosen = random.choice(low_deg_nodes)
                greedy_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
            # Local search on this greedy result
            ls_result = local_search_swap(greedy_set, deadline - 3.0)
            if len(ls_result) > len(best_solution):
                best_solution = ls_result

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
