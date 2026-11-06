import math
import time
import random

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before the 90s wall-clock kill

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

    # Precompute degrees for faster access
    deg = {a: len(conflict[a]) for a in candidates}

    # Maximum clique lower bound on conflict graph induced by `nodes`.
    # Uses greedy coloring on the conflict graph to find a large clique
    # (greedy coloring gives an upper bound on chromatic number, but we
    # use a branch-and-bound for max clique to get a lower bound on clique size).
    def max_clique_lower_bound(nodes, time_budget):
        """Return a lower bound on the maximum clique size in the conflict
        graph induced by `nodes`. Uses a greedy heuristic: repeatedly pick
        the vertex with highest degree in the remaining induced subgraph,
        add it to the clique, and remove it and its non-neighbors."""
        if not nodes:
            return 0
        if len(nodes) <= 4:
            # Brute force for small sets
            best = 0
            for r in range(1, len(nodes) + 1):
                # Check if any subset of size r is a clique
                # For very small sets, just check all
                pass
            # Simple: check if all are mutually connected
            all_connected = True
            nl = list(nodes)
            for i in range(len(nl)):
                for j in range(i + 1, len(nl)):
                    if nl[j] not in conflict[nl[i]]:
                        all_connected = False
                        break
                if not all_connected:
                    break
            if all_connected:
                return len(nodes)
            # Try pairs
            for i in range(len(nl)):
                for j in range(i + 1, len(nl)):
                    if nl[j] in conflict[nl[i]]:
                        best = max(best, 2)
            return best

        # Greedy clique finding: order by degree, greedily add
        rem = set(nodes)
        clique = []
        while rem:
            # Pick vertex with max degree in rem
            best_v = None
            best_d = -1
            for v in rem:
                d = len(conflict[v] & rem)
                if d > best_d:
                    best_d = d
                    best_v = v
            clique.append(best_v)
            rem.discard(best_v)
            # Keep only neighbors of best_v in rem
            rem &= conflict[best_v]
        return len(clique)

    # Upper bound on independence number of conflict graph induced by `nodes`:
    # alpha(G) <= |V| - omega(G)  where omega is max clique size
    # We use a lower bound on omega to get an upper bound on alpha.
    def independence_upper_bound(nodes, time_budget):
        if not nodes:
            return 0
        # Simple bound: number of vertices
        # Better: |nodes| - max_clique_lower_bound
        # But also: if graph is a matching, alpha = |nodes| - (number of edges)
        # Use greedy coloring as fallback (upper bound on alpha directly)
        # Actually, greedy coloring gives upper bound on alpha? No, it gives
        # upper bound on chromatic number, not directly on alpha.
        # The relation: alpha(G) * chi(G) >= |V|, so alpha(G) >= |V| / chi(G)
        # That's a LOWER bound on alpha, not upper.
        # For upper bound on alpha: alpha(G) <= |V| - omega(G)
        # So we need lower bound on omega(G).
        # Also: alpha(G) <= |V| / (min_degree + 1) by Turan-type bounds? Not exactly.
        # Let's use: alpha(G) <= |V| - max_clique_lower_bound(nodes)
        # And also the trivial bound |V|.
        # For the conflict graph which is sparse, max clique is small, so this bound
        # may not be much tighter than |V|.
        # Alternative: use the fact that if we can find a matching of size m,
        # then alpha <= |V| - m (each edge in matching forces at most 1 endpoint in IS).
        # Greedy matching:
        rem = set(nodes)
        matching_size = 0
        while rem:
            # Pick an edge
            found = False
            for u in list(rem):
                for v in conflict[u]:
                    if v in rem and v > u:  # avoid double counting
                        rem.discard(u)
                        rem.discard(v)
                        matching_size += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        # alpha <= |nodes| - matching_size
        bound1 = len(nodes) - matching_size
        # Also try clique bound
        # For sparse graphs, clique bound may be weak
        # Use the min of available bounds
        return bound1

    # Greedy coloring upper bound for independent set on a subgraph.
    # The number of colors used in a proper coloring of the CONFLICT graph
    # is an upper bound on the independence number? No.
    # Actually: alpha(G) <= |V| / chi(G) is false in general.
    # The correct relation: chi(G) * alpha(G) >= |V| only for some graphs.
    # For a proper coloring with k colors, each color class is an independent set,
    # so alpha(G) >= max color class size >= |V| / k.
    # That's a LOWER bound.
    # For UPPER bound: alpha(G) <= |V| - omega(G) is the standard one.
    # Or: if we find a vertex cover of size tau, then alpha = |V| - tau.
    # Greedy vertex cover:
    def vertex_cover_size(nodes):
        """Greedy vertex cover size (lower bound on min vertex cover)."""
        rem = set(nodes)
        cover = 0
        while rem:
            # Pick vertex with max degree
            best_v = None
            best_d = -1
            for v in rem:
                d = len(conflict[v] & rem)
                if d > best_d:
                    best_d = d
                    best_v = v
            if best_d == 0:
                break
            cover += 1
            rem.discard(best_v)
            rem -= conflict[best_v]
        return cover

    # Upper bound on alpha: |V| - vertex_cover_lower_bound
    # Since min vertex cover >= greedy vertex cover, and alpha = |V| - min_vc,
    # we have alpha <= |V| - greedy_vc.
    def independence_upper_bound_v2(nodes):
        if not nodes:
            return 0
        vc = vertex_cover_size(nodes)
        return len(nodes) - vc

    # Component decomposition for speed
    def _components(nodes):
        seen = set()
        comps = []
        for start_node in nodes:
            if start_node in seen:
                continue
            comp = set()
            stack = [start_node]
            seen.add(start_node)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in conflict[u]:
                    if v in nodes and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        return comps

    # Independent-set maximizer using branch-and-bound
    def maximize(pool, time_budget):
        pool = set(pool)
        # Greedy independent set (min-degree ordering) for a solid lower bound
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

        def search(sub, fixed):
            nonlocal best, blen
            if not sub:
                if len(fixed) > blen:
                    blen, best = len(fixed), set(fixed)
                return
            if time_up():
                return
            # Upper bound: use vertex cover based bound
            ub = len(fixed) + independence_upper_bound_v2(sub)
            if ub <= blen:
                return
            # Pick a high-degree vertex in sub to branch on
            v = None
            hv = -1
            for u in sub:
                d = len(conflict[u] & sub)
                if d > hv:
                    hv, v = d, u
            # Branch 1: include v
            sub_incl = (sub - {v}) - conflict[v]
            if len(fixed) + 1 + independence_upper_bound_v2(sub_incl) > blen:
                search(sub_incl, fixed | {v})
            # Branch 2: exclude v
            sub_excl = sub - {v}
            if len(fixed) + independence_upper_bound_v2(sub_excl) > blen:
                search(sub_excl, fixed)

        # Initial search over the whole pool
        search(pool, set())

        # Component decomposition: solve each connected component exactly
        comps = _components(pool)
        union = set()
        for c in comps:
            if time_up():
                break
            # Local exact solve on this component
            cbl = [0]
            cbest = [set()]

            # Greedy lower bound for this component
            rem_c = set(c)
            greedy_c = []
            while rem_c:
                best_c = None
                bd_c = None
                for node in rem_c:
                    d = len(conflict[node] & rem_c)
                    if bd_c is None or d < bd_c:
                        bd_c, best_c = d, node
                        if d == 0:
                            break
                greedy_c.append(best_c)
                rem_c.discard(best_c)
                rem_c -= conflict[best_c]
            cbl[0] = len(greedy_c)
            cbest[0] = set(greedy_c)

            def s2(sub, fixed):
                if not sub:
                    if len(fixed) > cbl[0]:
                        cbl[0] = len(fixed)
                        cbest[0] = set(fixed)
                    return
                if time.time() - start > time_budget:
                    return
                if len(fixed) + independence_upper_bound_v2(sub) <= cbl[0]:
                    return
                v = None
                hv = -1
                for u in sub:
                    d = len(conflict[u] & sub)
                    if d > hv:
                        hv, v = d, u
                sub_incl = (sub - {v}) - conflict[v]
                if len(fixed) + 1 + independence_upper_bound_v2(sub_incl) > cbl[0]:
                    s2(sub_incl, fixed | {v})
                sub_excl = sub - {v}
                if len(fixed) + independence_upper_bound_v2(sub_excl) > cbl[0]:
                    s2(sub_excl, fixed)

            s2(c, set())
            union |= cbest[0]

        if len(union) > blen:
            best, blen = set(union), len(union)
        return best

    def local_search(solution, time_budget):
        """Swap-based local search with fast 1-swap pass, then 1-removal greedy fill."""
        sol = set(solution)
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            # Pass 1: fast 1-swap (remove a, add b)
            for a in list(sol):
                if time.time() - start > time_budget:
                    return sol
                trial = sol - {a}
                for b in candidates:
                    if b in trial or b == a:
                        continue
                    if 2 * b in squares:
                        continue
                    if any(b + c in squares for c in trial):
                        continue
                    new_sol = trial | {b}
                    fill = set()
                    for c in candidates:
                        if c in new_sol:
                            continue
                        if 2 * c in squares:
                            continue
                        if all(c + d not in squares for d in (new_sol | fill)):
                            fill.add(c)
                    if len(new_sol) + len(fill) > len(sol):
                        sol = new_sol | fill
                        improved = True
                        break
                if improved:
                    break
            # Pass 2: 1-removal with greedy fill
            if not improved:
                for a in list(sol):
                    if time.time() - start > time_budget:
                        return sol
                    trial = sol - {a}
                    added = []
                    for b in candidates:
                        if b in trial:
                            continue
                        if 2 * b in squares:
                            continue
                        if all(b + c not in squares for c in (trial | set(added))):
                            added.append(b)
                    if len(trial) + len(added) > len(sol):
                        sol = trial | set(added)
                        improved = True
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

    # 3b. Final greedy-fill
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            best_solution.add(a)

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

    # 5. Safety verification
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
