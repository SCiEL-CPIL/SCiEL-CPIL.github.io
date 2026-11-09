"""
Maximize |S| for S ⊆ [1,512] with lcm(a,b) <= 512 for all a,b in S.

Strategy:
1. Build compatibility via bitmasks (fast).
2. Strong initial solution: divisors of highly-composite numbers + greedy + local add.
3. Exact branch-and-bound (Tomita-style) with greedy coloring upper bounds.
4. If B&B times out, do additional randomized local search with 2-for-1 swaps.
"""

def construct():
    import time
    import math
    import random

    start_time = time.time()
    LIMIT = 512
    N = LIMIT
    deadline = start_time + 82  # 8s safety margin out of 90s

    # ---- Build compatibility bitmasks ----
    # compat_mask[v]: bitmask of u in [1..N] such that lcm(v,u) <= LIMIT
    compat_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            l = (i // math.gcd(i, j)) * j
            if l <= LIMIT:
                mask |= (1 << j)
        compat_mask[i] = mask

    def is_valid(S):
        for i, a in enumerate(S):
            for b in S[i + 1:]:
                if not ((compat_mask[a] >> b) & 1):
                    return False
        return True

    def get_divisors(m):
        divs = []
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                divs.append(d)
                if d * d != m:
                    divs.append(m // d)
        return sorted(divs)

    # ---- Phase 1: Strong initial solution ----
    # Try divisors of several highly-composite numbers
    candidates_for_div = [480, 420, 360, 336, 240, 180, 120, 96, 72, 60, 48, 36, 24, 16, 12, 8, 6, 4, 2, 1]
    # Also find the number in [1,512] with the most divisors
    best_M = 1
    max_divs = 0
    for m in range(1, N + 1):
        cnt = len(get_divisors(m))
        if cnt > max_divs:
            max_divs = cnt
            best_M = m
    if best_M not in candidates_for_div:
        candidates_for_div.insert(0, best_M)

    def greedy_build(start_set):
        S_set = set(start_set)
        S_list = list(start_set)
        order = list(range(1, N + 1))
        order.sort(key=lambda x: -bin(compat_mask[x]).count('1'))
        for v in order:
            if v in S_set:
                continue
            if all((compat_mask[v] >> u) & 1 for u in S_list):
                S_set.add(v)
                S_list.append(v)
        return S_list

    def local_improve_add(S_list):
        S_set = set(S_list)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for v in range(1, N + 1):
                if v not in S_set and all((compat_mask[v] >> u) & 1 for u in S_set):
                    S_set.add(v)
                    S_list.append(v)
                    changed = True
        return S_list

    best_size = 0
    best_clique = []

    def update_best(S):
        nonlocal best_size, best_clique
        if is_valid(S) and len(S) > best_size:
            best_size = len(S)
            best_clique = list(S)

    # Try divisor sets
    for M in candidates_for_div:
        divs = [d for d in get_divisors(M) if d <= N]
        divs = local_improve_add(divs)
        update_best(divs)
        if time.time() > deadline - 60:
            break

    # Greedy from empty
    S = greedy_build([])
    S = local_improve_add(S)
    update_best(S)

    # ---- Phase 2: Exact branch-and-bound ----
    # adj_mask[v] = compat_mask[v] without self
    adj_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        adj_mask[v] = compat_mask[v] & ~(1 << v)

    node_count = 0

    def branch_and_bound(P, current):
        """
        P: list of candidate vertices that can still be added
        current: list of vertices in current clique
        """
        nonlocal best_size, best_clique, node_count
        node_count += 1

        if time.time() > deadline:
            return

        if not P:
            if len(current) > best_size:
                best_size = len(current)
                best_clique = list(current)
            return

        if len(current) + len(P) <= best_size:
            return

        # Greedy coloring of P for upper bound
        P_list = list(P)
        # Order by degree within P descending
        P_set = set(P_list)
        deg = {}
        for v in P_list:
            d = 0
            for u in P_list:
                if u != v and (compat_mask[v] >> u) & 1:
                    d += 1
            deg[v] = d
        P_list.sort(key=lambda x: -deg[x])

        color = {}
        max_color = 0
        for v in P_list:
            used = set()
            for u in P_list:
                if u != v and u in color and (compat_mask[v] >> u) & 1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c > max_color:
                max_color = c

        if len(current) + max_color <= best_size:
            return

        # Branch on vertices in reverse color order
        for v in reversed(P_list):
            if time.time() > deadline:
                return
            if len(current) + color[v] <= best_size:
                continue

            current.append(v)
            new_P = [u for u in P if u != v and (compat_mask[v] >> u) & 1]
            branch_and_bound(new_P, current)
            current.pop()

            # Remove v from P (move to excluded)
            # We don't explicitly track X; just shrink P
            # But we need to update P for remaining iterations
            # Actually in Tomita's algorithm, we remove v from P after branching
            # Let's restructure: iterate over a copy
            pass

        # The above loop doesn't properly remove v from P. Let's fix with a cleaner approach.
        # Actually, the standard Tomita approach iterates and removes. Let me redo.

    # Redo B&B with proper Tomita structure
    def branch_and_bound_v2(P, current):
        """
        P: list of candidate vertices
        current: list of vertices in current clique
        """
        nonlocal best_size, best_clique, node_count
        node_count += 1

        if time.time() > deadline:
            return

        if not P:
            if len(current) > best_size:
                best_size = len(current)
                best_clique = list(current)
            return

        if len(current) + len(P) <= best_size:
            return

        # Greedy coloring for upper bound
        P_list = list(P)
        P_set = set(P_list)
        deg = {}
        for v in P_list:
            d = 0
            for u in P_list:
                if u != v and (compat_mask[v] >> u) & 1:
                    d += 1
            deg[v] = d
        P_list.sort(key=lambda x: -deg[x])

        color = {}
        max_color = 0
        for v in P_list:
            used = set()
            for u in P_list:
                if u != v and u in color and (compat_mask[v] >> u) & 1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c > max_color:
                max_color = c

        if len(current) + max_color <= best_size:
            return

        # Branch in reverse order of P_list (higher color first)
        for idx in range(len(P_list) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = P_list[idx]
            if len(current) + color[v] <= best_size:
                continue

            # Include v
            current.append(v)
            new_P = [u for u in P_list[:idx] if (compat_mask[v] >> u) & 1]
            branch_and_bound_v2(new_P, current)
            current.pop()

    if time.time() < deadline - 5:
        full_P = list(range(1, N + 1))
        branch_and_bound_v2(full_P, [])

    # ---- Phase 3: If B&B didn't finish, do more local search ----
    if time.time() < deadline - 2 and best_clique:
        rng = random.Random(12345)

        # Try 2-for-1 swaps
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            S = list(best_clique)
            S_set = set(S)
            improved = False
            # Try removing one element and adding two
            for rem in S:
                new_set = S_set - {rem}
                new_list = [x for x in S if x != rem]
                # Find first addable
                add1 = None
                for v1 in range(1, N + 1):
                    if v1 not in new_set and all((compat_mask[v1] >> u) & 1 for u in new_set):
                        add1 = v1
                        break
                if add1 is None:
                    continue
                new_set2 = new_set | {add1}
                new_list2 = new_list + [add1]
                add2 = None
                for v2 in range(1, N + 1):
                    if v2 not in new_set2 and all((compat_mask[v2] >> u) & 1 for u in new_set2):
                        add2 = v2
                        break
                if add2 is not None:
                    new_set3 = new_set2 | {add2}
                    new_list3 = new_list2 + [add2]
                    if len(new_list3) > len(S):
                        S = new_list3
                        S_set = new_set3
                        improved = True
                        break
            if improved:
                update_best(S)
            else:
                break

        # Random restarts with perturbation
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            base = list(best_clique)
            if base:
                remove_k = min(len(base), rng.randint(1, max(1, len(base) // 4)))
                idxs = rng.sample(range(len(base)), remove_k)
                for i in sorted(idxs, reverse=True):
                    base.pop(i)
            S = greedy_build(base)
            S = local_improve_add(S)
            update_best(S)

    # Final validation
    if not best_clique:
        best_clique = [1]

    if not is_valid(best_clique):
        filtered = []
        for v in best_clique:
            if all((compat_mask[v] >> u) & 1 for u in filtered):
                filtered.append(v)
        best_clique = filtered if filtered else [1]

    return best_clique
