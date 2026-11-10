def construct():
    import time
    from math import gcd
    import random

    deadline = time.time() + 82.0

    N = 128
    all_numbers = list(range(1, N + 1))

    # Precompute GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    # Precompute conflict: for each pair (a,b), the set of x that form a bad triple
    # bad[x][a][b] would be too big; instead precompute for each pair (a,b) the list of x
    # But more useful: for each x and each pair (a,b), check quickly.
    # Let's precompute: for each x, for each pair (a,b) in [1..128], whether gcd(a,b)==gcd(a,x)==gcd(b,x)
    # That's 128 * 128 * 128 / 2 ≈ 1M entries, feasible.
    # Actually, let's precompute a 3D structure differently.
    # For each x, precompute gcd_with_x[a] = gcd(x, a) for all a.
    # Then for a pair (a,b), the check is: g[a][b] == gcd_with_x[a] == gcd_with_x[b]

    # Precompute gcd_with_x for each x
    gcd_with = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            gcd_with[x][a] = g[x][a]

    def can_add(x, S_list):
        """Check if x can be added to S_list without creating a violating triple."""
        n = len(S_list)
        gx = gcd_with[x]
        for i in range(n):
            a = S_list[i]
            gax = gx[a]
            for j in range(i + 1, n):
                b = S_list[j]
                if g[a][b] == gax and gx[b] == gax:
                    return False
        return True

    def conflict_count_for_x(x, S_list):
        """Count how many pairs in S_list would form a bad triple with x."""
        count = 0
        n = len(S_list)
        gx = gcd_with[x]
        for i in range(n):
            a = S_list[i]
            gax = gx[a]
            for j in range(i + 1, n):
                b = S_list[j]
                if g[a][b] == gax and gx[b] == gax:
                    count += 1
        return count

    def find_violations(S_list):
        """Find all violating triples in S_list."""
        violations = []
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                gab = g[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if gab == g[a][c] == g[b][c]:
                        violations.append((a, b, c))
        return violations

    def prune_violations(S_list, time_budget):
        """Remove elements to eliminate all violations."""
        S = set(S_list)
        S_list = list(S)
        while time.time() < time_budget:
            violations = find_violations(S_list)
            if not violations:
                return S_list
            conflict_count = {}
            for a, b, c in violations:
                conflict_count[a] = conflict_count.get(a, 0) + 1
                conflict_count[b] = conflict_count.get(b, 0) + 1
                conflict_count[c] = conflict_count.get(c, 0) + 1
            max_conf = -1
            to_remove = None
            for elem in S:
                cc = conflict_count.get(elem, 0)
                if cc > max_conf:
                    max_conf = cc
                    to_remove = elem
            if to_remove is None:
                break
            S.discard(to_remove)
            S_list = [x for x in S_list if x != to_remove]
        return S_list

    def greedy_add(order, S_init=None):
        S = set(S_init) if S_init else set()
        S_list = list(S)
        for x in order:
            if x in S:
                continue
            if can_add(x, S_list):
                S.add(x)
                S_list.append(x)
        return S_list

    def local_search_add(S_list, time_budget):
        S_set = set(S_list)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for x in all_numbers:
                if x not in S_set and can_add(x, S_list):
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
        return S_list

    def local_search_remove_add(S_list, time_budget):
        S_list = list(S_list)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for rem in S_list[:]:
                if time.time() > time_budget:
                    break
                S_trial = [x for x in S_list if x != rem]
                S_trial_set = set(S_trial)
                candidates = [x for x in all_numbers if x not in S_trial_set]
                for x in candidates:
                    if can_add(x, S_trial):
                        new_S = S_trial + [x]
                        for z in candidates:
                            if z == x:
                                continue
                            if can_add(z, new_S):
                                S_list = new_S + [z]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return S_list

    def swap_improve(S_list, time_budget):
        S = set(S_list)
        S_list = list(S)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for rem in S_list[:]:
                if time.time() > time_budget:
                    break
                S_trial = S - {rem}
                S_trial_list = [x for x in S_list if x != rem]
                candidates = [x for x in all_numbers if x not in S_trial]
                if len(candidates) <= 80:
                    candidates.sort(key=lambda x: conflict_count_for_x(x, S_trial_list))
                else:
                    candidates.sort(key=lambda x: x)
                added_any = False
                for x in candidates:
                    if time.time() > time_budget:
                        break
                    if can_add(x, S_trial_list):
                        S_trial.add(x)
                        S_trial_list.append(x)
                        added_any = True
                if len(S_trial) > len(S):
                    S = S_trial
                    S_list = list(S)
                    improved = True
                    break
        return S_list

    # Branch and bound
    def branch_and_bound(S_current, unselected, time_budget, best_size):
        S = set(S_current)
        S_list = list(S)
        unselected_set = set(unselected)
        best = list(S_list)
        best_size_local = len(S_list)

        def compute_degree(x, S_list, unselected_list):
            deg = 0
            gx = gcd_with[x]
            for i in range(len(S_list)):
                a = S_list[i]
                gax = gx[a]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    if g[a][b] == gax and gx[b] == gax:
                        deg += 1
            ul = unselected_list
            if len(ul) <= 40:
                for i in range(len(ul)):
                    a = ul[i]
                    gax = gx[a]
                    for j in range(i + 1, len(ul)):
                        b = ul[j]
                        if g[a][b] == gax and gx[b] == gax:
                            deg += 1
            return deg

        def bnb_recursive(S_list, unselected_list, depth, time_budget):
            nonlocal best, best_size_local
            if time.time() > time_budget:
                return
            if not unselected_list:
                if len(S_list) > best_size_local:
                    best_size_local = len(S_list)
                    best = list(S_list)
                return
            if len(S_list) + len(unselected_list) <= best_size_local:
                return

            if len(unselected_list) <= 30:
                degrees = [(compute_degree(x, S_list, unselected_list), x) for x in unselected_list]
                degrees.sort(key=lambda t: -t[0])
                ordered = [x for _, x in degrees]
            else:
                ordered = sorted(unselected_list, key=lambda x: x)

            x = ordered[0]
            if can_add(x, S_list):
                bnb_recursive(S_list + [x], ordered[1:], depth + 1, time_budget)

            bnb_recursive(S_list, ordered[1:], depth + 1, time_budget)

        if len(unselected) <= 50 and time.time() < time_budget - 5:
            bnb_recursive(S_list, list(unselected), 0, time_budget)

        if len(best) > best_size:
            return best
        return S_list

    # === Main search ===
    best = []

    random.seed(42)

    def num_divisors(n):
        count = 0
        for d in range(1, int(n ** 0.5) + 1):
            if n % d == 0:
                count += 2
                if d * d == n:
                    count -= 1
        return count

    def num_distinct_pf(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    div_counts = {x: num_divisors(x) for x in all_numbers}
    pf_counts = {x: num_distinct_pf(x) for x in all_numbers}

    orderings = []
    orderings.append(sorted(all_numbers, key=lambda x: (div_counts[x], x)))
    orderings.append(sorted(all_numbers, key=lambda x: (-pf_counts[x], x)))
    orderings.append(list(all_numbers))
    orderings.append(list(reversed(all_numbers)))
    for i in range(50):
        order = all_numbers[:]
        random.seed(i * 7919 + 13)
        random.shuffle(order)
        orderings.append(order)

    for order in orderings:
        if time.time() > deadline - 15:
            break
        S = greedy_add(order)
        if len(S) > len(best):
            best = S

    dense_starts = [
        list(range(1, N + 1)),
        [x for x in range(1, N + 1) if x % 2 == 1],
        [x for x in range(1, N + 1) if x % 2 == 0],
        [x for x in range(1, N + 1) if x % 3 != 0],
        [x for x in range(1, N + 1) if x % 5 != 0],
        [x for x in range(1, N + 1) if x % 2 != 0 or x % 3 != 0],
    ]

    for ds in dense_starts:
        if time.time() > deadline - 15:
            break
        S = prune_violations(ds, deadline - 10)
        S = local_search_add(S, deadline - 10)
        if time.time() < deadline - 8:
            S = swap_improve(S, deadline - 8)
        if len(S) > len(best):
            best = S

    if time.time() < deadline - 10:
        best = local_search_add(best, deadline - 10)
        best = local_search_remove_add(best, deadline - 8)
        if time.time() < deadline - 5:
            best = swap_improve(best, deadline - 5)

    for iteration in range(30):
        if time.time() > deadline - 5:
            break
        S_perturbed = set(best)
        S_list_perturbed = list(S_perturbed)
        k = min(3, len(S_list_perturbed))
        to_remove = random.sample(S_list_perturbed, k)
        for x in to_remove:
            S_perturbed.discard(x)
        S_perturbed_list = list(S_perturbed)

        S_opt = local_search_add(S_perturbed_list, deadline - 5)
        if time.time() < deadline - 3:
            S_opt = swap_improve(S_opt, deadline - 3)
        if len(S_opt) > len(best):
            best = S_opt

    if time.time() < deadline - 3:
        best_set = set(best)
        unselected = [x for x in all_numbers if x not in best_set]
        if len(unselected) <= 50:
            bnb_result = branch_and_bound(best, unselected, deadline - 2, len(best))
            if len(bnb_result) > len(best):
                best = bnb_result

    best = prune_violations(best, deadline - 1)
    best = local_search_add(best, deadline - 1)

    if not best:
        best = [1]

    return best
