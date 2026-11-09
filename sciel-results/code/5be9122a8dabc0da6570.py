"""
Maximize |S| for S ⊆ [1,512] with lcm(a,b) <= 512 for all a,b in S.

Strategy:
1. Build a 512x512 compatibility matrix.
2. Use a MaxSAT-style greedy local search that builds a large valid set.
3. Run many randomized restarts + greedy + local-improvement sweeps.
4. Validate and return the best set found.
"""

def construct():
    import time
    import math
    import random
    from itertools import combinations

    start_time = time.time()
    LIMIT = 512
    N = LIMIT
    deadline = start_time + 85  # 5 s safety margin out of 90 s

    # ---- Build compatibility: lcm(i, j) <= 512 ----
    # For speed, precompute lcm table using math.gcd
    # compat[v][w] for v, w in 1..512 (1-indexed)
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = (i // math.gcd(i, j)) * j
            if l <= LIMIT:
                compat[i][j] = True
                compat[j][i] = True

    # Helper: check if a set is valid (lcm of all pairs <= LIMIT)
    def is_valid(S):
        for i, a in enumerate(S):
            for b in S[i + 1:]:
                if not compat[a][b]:
                    return False
        return True

    def count_violations(S):
        c = 0
        for i, a in enumerate(S):
            for b in S[i + 1:]:
                if not compat[a][b]:
                    c += 1
        return c

    def valid_addable_to(S_set, S_list):
        """Yield v in [1..N] \ S such that adding v keeps the set valid."""
        for v in range(1, N + 1):
            if v not in S_set:
                if all(compat[v][u] for u in S_list):
                    yield v

    # ---- Greedy builder from a starting set ----
    def greedy_build(start_set, random_order=False, seed=None):
        S_set = set(start_set)
        S_list = list(start_set)
        if random_order:
            rng = random.Random(seed)
            order = list(range(1, N + 1))
            rng.shuffle(order)
        else:
            # Order by "ease of compatibility" = number of compatibles descending
            order = list(range(1, N + 1))
            order.sort(key=lambda x: -sum(compat[x][y] for y in range(1, N + 1) if y != x))
        for v in order:
            if v in S_set:
                continue
            if all(compat[v][u] for u in S_list):
                S_set.add(v)
                S_list.append(v)
            if time.time() > deadline:
                break
        return S_list

    # ---- Local improvement: add/remove to maximize |S| while valid ----
    def local_improve(S_list, max_iter=300, random_seed=None):
        if random_seed is None:
            rng = random.Random()
        else:
            rng = random.Random(random_seed)
        S_set = set(S_list)
        # Phase 1: try to add nodes
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < deadline:
            improved = False
            it += 1
            candidates = [v for v in range(1, N + 1) if v not in S_set]
            rng.shuffle(candidates)
            for v in candidates:
                if all(compat[v][u] for u in S_set):
                    S_set.add(v)
                    S_list.append(v)
                    improved = True
        # Phase 2: try remove-one-add-two swaps (2-for-1)
        improved = True
        while improved and it < max_iter * 2 and time.time() < deadline:
            improved = False
            it += 1
            current_list = list(S_set)
            rng.shuffle(current_list)
            for rem in current_list:
                # Tentatively remove rem
                new_set = S_set - {rem}
                new_list = [x for x in S_list if x != rem]
                # Try to add two new nodes
                add1 = None
                for v1 in range(1, N + 1):
                    if v1 not in new_set and all(compat[v1][u] for u in new_set):
                        add1 = v1
                        break
                if add1 is None:
                    continue
                new_set2 = new_set | {add1}
                new_list2 = new_list + [add1]
                add2 = None
                for v2 in range(1, N + 1):
                    if v2 not in new_set2 and all(compat[v2][u] for u in new_set2):
                        add2 = v2
                        break
                if add2 is not None:
                    S_set = new_set2 | {add2}
                    S_list = new_list2 + [add2]
                    improved = True
                    break
        return S_list

    # ---- Divisor-set strategy: find a number M <= 512 with many divisors ----
    # All divisors of M have pairwise lcm dividing M <= 512, so they form a valid set.
    def get_divisors(m):
        divs = []
        for d in range(1, m + 1):
            if m % d == 0:
                divs.append(d)
        return divs

    # Find the number in [1,512] with the most divisors
    best_M = 1
    max_divs = 0
    for m in range(1, N + 1):
        cnt = 0
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                cnt += 1
                if d * d != m:
                    cnt += 1
        if cnt > max_divs:
            max_divs = cnt
            best_M = m

    divisor_set = get_divisors(best_M)
    # All divisors are <= best_M <= 512, so valid subset of [1,512]
    divisor_set = [d for d in divisor_set if d <= N]

    # ---- Search: many varied starts + local improvement ----
    rng = random.Random(42)
    best_set = []
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        # Validate
        if is_valid(S) and len(S) > best_size:
            best_set = list(S)
            best_size = len(S)

    # Start 1: divisor set as seed (known good baseline)
    update_best(divisor_set)

    # Start 2: empty start
    S = greedy_build([])
    update_best(S)

    # Start 3: divisor set + greedy to grow
    S = greedy_build(divisor_set)
    update_best(S)

    # Start 4: singletons of high-degree nodes
    degree = [sum(compat[v][u] for u in range(1, N + 1) if u != v) for v in range(N + 1)]
    top_nodes = sorted(range(1, N + 1), key=lambda v: -degree[v])[:40]
    for v in top_nodes:
        if time.time() > deadline:
            break
        S = greedy_build([v])
        update_best(S)

    # Start 5: random small subsets as seeds
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        seed_size = rng.randint(1, 6)
        seed = rng.sample(range(1, N + 1), seed_size)
        S = greedy_build(seed)
        update_best(S)

    # Start 6: divisor set perturbed + greedy
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        perturbed_divs = divisor_set[:]
        # Randomly remove a few divisors
        if perturbed_divs:
            remove_k = min(len(perturbed_divs), rng.randint(1, max(1, len(perturbed_divs) // 5)))
            idxs = rng.sample(range(len(perturbed_divs)), remove_k)
            for i in sorted(idxs, reverse=True):
                perturbed_divs.pop(i)
        S = greedy_build(perturbed_divs)
        update_best(S)

    # Local improvement on the best found so far, with several random variants
    if best_set:
        # Several random local-improvement passes on permutations/perturbations of best_set
        base = list(best_set)
        for k in range(50):
            if time.time() > deadline - 2:
                break
            perturbed = base[:]
            # Randomly remove a few
            if perturbed:
                remove_k = min(len(perturbed), rng.randint(1, max(1, len(perturbed) // 5)))
                idxs = rng.sample(range(len(perturbed)), remove_k)
                for i in sorted(idxs, reverse=True):
                    perturbed.pop(i)
            S = local_improve(perturbed, max_iter=100, random_seed=k)
            update_best(S)

    # Re-run greedy with different orderings on the current best to see if we can grow
    if best_set:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            base = list(best_set)
            # Randomly remove 0-3 nodes
            if base:
                remove_k = min(len(base), rng.randint(0, 3))
                if remove_k > 0:
                    idxs = rng.sample(range(len(base)), remove_k)
                    for i in sorted(idxs, reverse=True):
                        base.pop(i)
            S = greedy_build(base, random_order=True, seed=_ * 12345)
            update_best(S)

    # Final: exhaustive small-neighborhood search around best
    if best_set and time.time() < deadline - 2:
        base = list(best_set)
        base_set = set(base)
        # Try every single addition
        for v in range(1, N + 1):
            if v in base_set:
                continue
            if all(compat[v][u] for u in base):
                newS = base + [v]
                update_best(newS)

    # Aggressive local improvement: try 2-for-1 and 3-for-2 swaps more thoroughly
    if best_set and time.time() < deadline - 2:
        base = list(best_set)
        base_set = set(base)
        # Try removing one element and adding two
        for rem in base:
            if time.time() > deadline - 2:
                break
            new_set = base_set - {rem}
            new_list = [x for x in base if x != rem]
            # Find all addable elements
            addable = [v for v in range(1, N + 1) if v not in new_set and all(compat[v][u] for u in new_list)]
            if len(addable) >= 2:
                # Try all pairs of addable elements
                for a1, a2 in combinations(addable, 2):
                    if compat[a1][a2] and all(compat[a1][u] for u in new_list) and all(compat[a2][u] for u in new_list):
                        newS = new_list + [a1, a2]
                        update_best(newS)
                        break
            elif len(addable) == 1:
                newS = new_list + addable
                if len(newS) > len(base):
                    update_best(newS)

    # If best is empty (shouldn't happen), fallback to divisor set
    if not best_set:
        best_set = divisor_set if divisor_set else [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Final validation
    if not is_valid(best_set):
        # Filter
        filtered = []
        for v in best_set:
            if all(compat[v][u] for u in filtered):
                filtered.append(v)
        best_set = filtered

    return best_set
