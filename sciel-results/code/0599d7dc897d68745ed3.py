import math
import time
import random
import numpy as np

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
    M = len(candidates)
    node_to_idx = {v: i for i, v in enumerate(candidates)}

    # Conflicts restricted to the candidate set.
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    # Greedy coloring upper bound for independent set on a subgraph.
    def color_bound(nodes):
        if not nodes:
            return 0
        degs = [(len(conflict[u] & nodes), u) for u in nodes]
        degs.sort(reverse=True)
        ordered = [u for _, u in degs]
        color_of = {}
        next_color = 0
        for u in ordered:
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

    def time_up(budget=None):
        if budget is None:
            return time.time() - start > deadline - 3.0
        return time.time() - start > budget

    def try_add_fast(s, v):
        return len(conflict[v] & s) == 0

    # --- MILP solve for a strong initial solution (and upper bound) ---
    best_solution = None
    milp_upper = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        # Build constraint matrix
        rows = []
        for idx, a in enumerate(candidates):
            for b in conflict[a]:
                if node_to_idx[b] > idx:
                    row = np.zeros(M)
                    row[idx] = 1.0
                    row[node_to_idx[b]] = 1.0
                    rows.append(row)
        if rows:
            A = np.array(rows)
            b_ub = np.ones(len(rows))
            c = -np.ones(M)
            bounds_obj = Bounds(lb=0, ub=1)
            integrality = np.ones(M)
            constraints = LinearConstraint(A, ub=b_ub)
            res = milp(c=c, constraints=constraints, bounds=bounds_obj, integrality=integrality)
            if res.status == 0 and res.x is not None:
                selected = [candidates[i] for i in range(M) if res.x[i] > 0.5]
                best_solution = set(selected)
                # Upper bound from LP relaxation value
                milp_upper = int(-res.fun) if res.fun is not None else None
    except Exception:
        pass

    # If MILP didn't give a solution, use greedy
    if best_solution is None:
        rem = set(candidates)
        greedy = set()
        while rem:
            best = None
            bd = None
            for node in rem:
                d = len(conflict[node] & rem)
                if bd is None or d < bd:
                    bd, best = d, node
                    if d == 0:
                        break
            greedy.add(best)
            rem.discard(best)
            rem -= conflict[best]
        best_solution = greedy

    # --- Local search refinement ---
    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                if time_up(time_budget):
                    break
                s.remove(u)
                addable = []
                for v in candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved_flag = True
                    break
                elif len(addable) == 1:
                    deg_u = len(conflict[u] & s)
                    deg_v = len(conflict[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved_flag = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

        return s

    # Refine the initial solution with local search
    best_solution = local_search_improve(best_solution, deadline - 25.0)

    # --- Branch-and-bound on components for further improvement ---
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

        def t_up():
            return time.time() - start > time_budget

        def search(sub, fixed):
            nonlocal best, blen
            if not sub:
                if len(fixed) > blen:
                    blen, best = len(fixed), set(fixed)
                return
            if t_up():
                return
            ub = len(fixed) + color_bound(sub)
            if ub <= blen:
                return
            v = None
            hv = -1
            for u in sub:
                d = len(conflict[u] & sub)
                if d > hv:
                    hv, v = d, u
            sub_include = (sub - {v}) - conflict[v]
            if len(fixed) + 1 + color_bound(sub_include) > blen:
                search(sub_include, fixed | {v})
            sub_exclude = sub - {v}
            if len(fixed) + color_bound(sub_exclude) > blen:
                search(sub_exclude, fixed)

        search(pool, set())

        # component decomposition
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
            if t_up():
                break
        union = set()
        for c in comps:
            sub_best = set()
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
            if t_up():
                break
        if len(union) > blen:
            best, blen = set(union), len(union)
        return best

    # Run branch-and-bound on the full candidate pool, seeded with current best
    sol = maximize(candidates, deadline - 15.0)
    if len(sol) > len(best_solution):
        best_solution = set(sol)
    # Also try local search on the B&B result
    ls_result = local_search_improve(best_solution, deadline - 10.0)
    if len(ls_result) > len(best_solution):
        best_solution = ls_result

    # --- Random restarts with different orderings ---
    if not time_up(deadline - 10.0):
        for restart in range(100):
            if time_up(deadline - 5.0):
                break
            shuffled = candidates[:]
            random.shuffle(shuffled)
            current_set = set()
            for v in shuffled:
                if try_add_fast(current_set, v):
                    current_set.add(v)
            ls_result = local_search_improve(current_set, deadline - 5.0)
            if len(ls_result) > len(best_solution):
                best_solution = ls_result

    # --- Final polish: try to add any remaining node ---
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            best_solution.add(a)

    # Safety verification: ensure no pair (a<=b) has a square sum.
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
