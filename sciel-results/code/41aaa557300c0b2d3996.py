# EVOLVE-BLOCK-START
def construct():
    import math
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    def lcm(a, b):
        return a * b // math.gcd(a, b)

    n = 128
    universe = list(range(1, n + 1))

    # Precompute LCM for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[(i, j)] = lcm(i, j)

    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a, b, c in combinations(range(1, n + 1), 3):
        if lcm_table[(a, b)] == lcm_table[(a, c)] == lcm_table[(b, c)]:
            bad_triples.append((a, b, c))

    # For fast checking: for each pair, list of third elements that form bad triple
    pair_to_thirds = {}
    for (a, b, c) in bad_triples:
        for pair in [(a, b), (a, c), (b, c)]:
            pair_to_thirds.setdefault(pair, set()).add(c if pair == (a, b) else (b if pair == (a, c) else a))
        # Simpler: rebuild properly
    pair_to_thirds = {}
    for (a, b, c) in bad_triples:
        pair_to_thirds.setdefault((a, b), set()).add(c)
        pair_to_thirds.setdefault((a, c), set()).add(b)
        pair_to_thirds.setdefault((b, c), set()).add(a)

    def is_valid_subset(S):
        S = set(S)
        for (a, b) in combinations(sorted(S), 2):
            if pair_to_thirds.get((a, b), set()) & S:
                return False
        return True

    def count_bad(S):
        S = set(S)
        cnt = 0
        for (a, b) in combinations(sorted(S), 2):
            cnt += len(pair_to_thirds.get((a, b), set()) & S)
        return cnt

    # Greedy construction with different orderings
    def greedy(order):
        selected = []
        sel_set = set()
        for x in order:
            if all(not (pair_to_thirds.get((a, x), set()) & sel_set or pair_to_thirds.get((x, a), set()) & sel_set) for a in selected):
                # check all pairs in selected with x
                ok = True
                for a in selected:
                    key = (a, x) if a < x else (x, a)
                    if pair_to_thirds.get(key, set()) & sel_set:
                        ok = False
                        break
                if ok:
                    selected.append(x)
                    sel_set.add(x)
        return selected

    def local_search(S, max_iters=20000):
        S = list(S)
        S_set = set(S)
        best = len(S)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try to add each excluded element
            excluded = [x for x in universe if x not in S_set]
            random.shuffle(excluded)
            for x in excluded:
                if time.time() > DEADLINE:
                    break
                # check if adding x creates any bad triple
                ok = True
                for a in S:
                    key = (a, x) if a < x else (x, a)
                    if pair_to_thirds.get(key, set()) & S_set:
                        ok = False
                        break
                if ok:
                    S.append(x)
                    S_set.add(x)
                    best = len(S)
                    improved = True
            # Try swaps: remove one, add one or two
            if time.time() > DEADLINE:
                break
            for i in range(len(S)):
                if time.time() > DEADLINE:
                    break
                a = S[i]
                S_set.remove(a)
                # try adding any two excluded
                excluded = [x for x in universe if x not in S_set]
                random.shuffle(excluded)
                for x in excluded:
                    if time.time() > DEADLINE:
                        break
                    ok = True
                    for b in S:
                        key = (b, x) if b < x else (x, b)
                        if pair_to_thirds.get(key, set()) & S_set:
                            ok = False
                            break
                    if ok:
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best:
                            best = len(S)
                        improved = True
                        break
                # try adding one
                if not improved or time.time() > DEADLINE:
                    excluded = [x for x in universe if x not in S_set]
                    random.shuffle(excluded)
                    for x in excluded:
                        if time.time() > DEADLINE:
                            break
                        ok = True
                        for b in S:
                            key = (b, x) if b < x else (x, b)
                            if pair_to_thirds.get(key, set()) & S_set:
                                ok = False
                                break
                        if ok:
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best:
                                best = len(S)
                            improved = True
                            break
                S_set.add(a)
        return S

    # Multiple greedy starts with different orderings
    best_solution = []
    best_len = 0

    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
    ]
    # Add a few random orderings
    for _ in range(3):
        o = list(range(1, n + 1))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        if time.time() > DEADLINE - 10:
            break
        S = greedy(order)
        if len(S) > best_len:
            best_len = len(S)
            best_solution = S[:]
        # Local search
        S2 = local_search(S)
        if len(S2) > best_len:
            best_len = len(S2)
            best_solution = S2[:]

    # Final verification
    if not best_solution:
        best_solution = [1]

    return best_solution
# EVOLVE-BLOCK-END
