"""Combined program: hypergraph adjacency + ILP polish + enhanced local search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def remaining():
        return DEADLINE - (time.time() - t0)

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def is_sum_free(S):
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def build_adjacency(elements):
        """Build hypergraph adjacency: adj[v] = list of (u, w) such that v+u=w or v+w=u or u+w=v."""
        V_set = set(elements)
        V_list = list(V_set)
        n = len(V_list)
        adj = {v: [] for v in V_set}
        for i, a in enumerate(V_list):
            # 2a case
            if 2 * a in V_set:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            for j in range(i + 1, n):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    adj[a].append((b, c))
                    adj[b].append((a, c))
                    adj[c].append((a, b))
        return adj, V_set

    def check_conflict(v, S_set, adj):
        """Check if v can be added to S_set using precomputed adjacency."""
        for (u, w) in adj[v]:
            if u == v:
                if w in S_set:
                    return False
            else:
                if u in S_set and w in S_set:
                    return False
        return True

    def greedy_fill(order, adj, V_set):
        """Greedy fill following given order using adjacency check."""
        S = set()
        for v in order:
            if v not in S and check_conflict(v, S, adj):
                S.add(v)
        return S

    def local_search_add(S_init, adj, V_set, time_budget, rng):
        """Hill-climbing: try adding elements, with 1-swap and 2-swap moves."""
        S = set(S_init)
        best_S = set(S)
        best_size = len(S)
        not_in = list(V_set - S)
        no_improve = 0
        max_no_improve = 20000

        # Conflict count for not_in elements
        def compute_conflicts(not_in_list, cur_set):
            cc = {}
            for x in not_in_list:
                cnt = 0
                for (u, w) in adj[x]:
                    if u == x:
                        if w in cur_set:
                            cnt += 1
                    else:
                        if u in cur_set and w in cur_set:
                            cnt += 1
                cc[x] = cnt
            return cc

        conflict_count = compute_conflicts(not_in, S)

        while time_budget() > 0.3 and no_improve < max_no_improve:
            if not not_in:
                break

            # Try to find a zero-conflict element
            zero_conflict = [x for x in not_in if conflict_count.get(x, 0) == 0]
            if zero_conflict:
                x = min(zero_conflict, key=lambda v: abs(v))
                S.add(x)
                not_in.remove(x)
                no_improve = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                # Update conflict counts
                for y in not_in:
                    for (u, w) in adj[y]:
                        if u == y:
                            if w == x:
                                conflict_count[y] = conflict_count.get(y, 0) + 1
                        else:
                            if (u == x and w in S) or (w == x and u in S):
                                conflict_count[y] = conflict_count.get(y, 0) + 1
                continue

            # No zero-conflict: try 1-swap (remove one, add one)
            x = rng.choice(not_in)
            # Find conflicts for x
            conflicts = []
            for (u, w) in adj[x]:
                if u == x:
                    if w in S:
                        conflicts.append((w, w))
                else:
                    if u in S and w in S:
                        conflicts.append((u, w))

            if not conflicts:
                S.add(x)
                not_in.remove(x)
                no_improve = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue

            # Find which elements in S cause conflicts
            freq = {}
            for (u, w) in conflicts:
                if u == w:
                    freq[u] = freq.get(u, 0) + 1
                else:
                    freq[u] = freq.get(u, 0) + 1
                    freq[w] = freq.get(w, 0) + 1

            if not freq:
                no_improve += 1
                continue

            # Try removing the most-conflicting element
            max_f = max(freq.values())
            candidates_rem = [k for k, c in freq.items() if c == max_f]
            rem = rng.choice(candidates_rem)

            # 1-swap: remove rem, add x
            S.discard(rem)
            S.add(x)
            if rem in not_in:
                pass
            if x in not_in:
                not_in.remove(x)
            not_in.append(rem)

            # Verify
            valid = True
            for (u, w) in adj[x]:
                if u == x:
                    if w in S:
                        valid = False
                        break
                else:
                    if u in S and w in S:
                        valid = False
                        break

            if not valid:
                # Revert
                S.discard(x)
                S.add(rem)
                if rem in not_in:
                    not_in.remove(rem)
                not_in.append(x)
                no_improve += 1
            else:
                # Try 2-swap: remove two, add x (only if still not adding net)
                # Actually after 1-swap size is same. Try to also add another.
                no_improve = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)

                # Update conflict counts
                conflict_count = compute_conflicts(not_in, S)

        return best_S

    def ilp_polish(S_init, elements, adj, V_set, time_budget):
        """Use ILP to find maximum sum-free superset of S_init within remaining elements."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
        except ImportError:
            return set(S_init)

        t_start = time.time()
        S = set(S_init)
        not_in = sorted(V_set - S)
        if not not_in:
            return S

        m = len(not_in)
        # Variables: x_i = 1 if we add not_in[i]
        # Constraint: for each hyperedge (a,b,c), not all three in S ∪ {added}
        # If a,b,c all in S: already violated (shouldn't happen)
        # If two in S, one in not_in: that not_in element cannot be added
        # If one in S, two in not_in: x_i + x_j <= 1
        # If zero in S, three in not_in: x_i + x_j + x_k <= 2

        # Map not_in elements to indices
        idx_map = {v: i for i, v in enumerate(not_in)}

        # Build constraints
        A_rows = []
        b_ub = []

        # For each element in not_in, check its adjacency
        for v in not_in:
            vi = idx_map[v]
            for (u, w) in adj[v]:
                if u == v:
                    # 2v = w or v = w/2
                    if w in S:
                        # v cannot be added at all
                        # x_v = 0
                        A_rows.append([0] * m)
                        # Set x_v <= 0 via bounds
                        # Handle via bounds
                        continue
                    elif w in idx_map:
                        # x_v + x_w <= 1
                        row = [0] * m
                        row[vi] = 1
                        row[idx_map[w]] = 1
                        A_rows.append(row)
                        b_ub.append(1)
                    else:
                        # w not in S and not in not_in: no constraint
                        pass
                else:
                    # v + u = w or v + w = u
                    u_in_S = u in S
                    w_in_S = w in S
                    u_in_ni = u in idx_map
                    w_in_ni = w in idx_map

                    if u_in_S and w_in_S:
                        # v cannot be added
                        A_rows.append([0] * m)
                        continue
                    elif u_in_S and w_in_ni:
                        # x_v + x_w <= 1
                        row = [0] * m
                        row[vi] = 1
                        row[idx_map[w]] = 1
                        A_rows.append(row)
                        b_ub.append(1)
                    elif w_in_S and u_in_ni:
                        # x_v + x_u <= 1
                        row = [0] * m
                        row[vi] = 1
                        row[idx_map[u]] = 1
                        A_rows.append(row)
                        b_ub.append(1)
                    elif u_in_ni and w_in_ni:
                        # x_v + x_u + x_w <= 2
                        row = [0] * m
                        row[vi] = 1
                        row[idx_map[u]] = 1
                        row[idx_map[w]] = 1
                        A_rows.append(row)
                        b_ub.append(2)

        if not A_rows:
            # No constraints, add all
            S.update(not_in)
            return S

        A = np.array(A_rows, dtype=float)
        c = np.ones(m)  # Maximize sum
        bounds = Bounds(lb=0, ub=1)
        constraints = [LinearConstraint(A, ub=b_ub)]

        try:
            res = milp(c=c, constraints=constraints, bounds=bounds,
                       integrality=np.ones(m))
            if res.status == 0:
                for i, v in enumerate(not_in):
                    if res.x[i] > 0.5:
                        S.add(v)
        except Exception:
            pass

        return S

    def solve_instance(arr, budget):
        """Solve one instance with hypergraph adjacency + local search + ILP polish."""
        t_start = time.time()
        adj, V_set = build_adjacency(arr)
        n = len(V_set)

        # --- Multiple greedy initializations ---
        candidates = []

        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        candidates.append(S_odds)

        # 2. Greedy by absolute value ascending
        order_abs = sorted(V_set, key=lambda x: abs(x))
        candidates.append(greedy_fill(order_abs, adj, V_set))

        # 3. Greedy by absolute value descending
        order_abs_desc = sorted(V_set, key=lambda x: -abs(x))
        candidates.append(greedy_fill(order_abs_desc, adj, V_set))

        # 4. Greedy by value ascending
        order_val = sorted(V_set, key=lambda x: x)
        candidates.append(greedy_fill(order_val, adj, V_set))

        # 5. Greedy by value descending
        order_val_desc = sorted(V_set, key=lambda x: -x)
        candidates.append(greedy_fill(order_val_desc, adj, V_set))

        # 6-10. Random orderings
        rng_init = random.Random(42)
        for _ in range(5):
            order = list(V_set)
            rng_init.shuffle(order)
            candidates.append(greedy_fill(order, adj, V_set))

        # Pick best
        best_S = max(candidates, key=len)
        best_size = len(best_S)

        # --- Local search from best initial ---
        ls_budget = budget * 0.5
        rng_ls = random.Random(999)
        improved = local_search_add(best_S, adj, V_set, lambda: ls_budget - (time.time() - t_start), rng_ls)
        if len(improved) > best_size:
            best_S = improved
            best_size = len(best_S)

        # --- Random restarts ---
        t_restart = time.time()
        restart_budget = budget * 0.3
        rng_restart = random.Random(12345)
        guard = 0
        while (time.time() - t_restart) < restart_budget and guard < 30:
            guard += 1
            order = list(V_set)
            rng_restart.shuffle(order)
            S_init = greedy_fill(order, adj, V_set)
            S_ls = local_search_add(S_init, adj, V_set, lambda: budget * 0.03, random.Random(guard))
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_S = S_ls

        # --- ILP polish ---
        ilp_budget = max(1.0, budget * 0.2)
        if remaining() > ilp_budget + 2.0:
            polished = ilp_polish(best_S, arr, adj, V_set, ilp_budget)
            if len(polished) > best_size:
                best_S = polished
                best_size = len(best_S)

        # Final safety check
        if not is_sum_free(list(best_S)):
            best_S = greedy_fill(sorted(V_set, key=lambda x: abs(x)), adj, V_set)

        return list(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = gen_instance(size, seed)
        elapsed = time.time() - t0
        remaining_time = DEADLINE - elapsed - 5.0
        if remaining_time < 3:
            # Emergency: return odds
            results.append([x for x in arr if x % 2 != 0])
            continue
        # Allocate budget based on instance size
        if size <= 60:
            budget = min(15.0, remaining_time)
        elif size <= 120:
            budget = min(35.0, remaining_time)
        else:
            budget = min(70.0, remaining_time)
        res = solve_instance(arr, budget)
        results.append(res)

    return results
# EVOLVE-BLOCK-END
