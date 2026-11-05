import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    numbers = list(range(1, n + 1))

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Precompute all forbidden triples
    forbidden_triples = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lab = get_lcm(i, j)
            for k in range(j + 1, n + 1):
                if get_lcm(i, k) == lab and get_lcm(j, k) == lab:
                    forbidden_triples.append((i, j, k))

    # Build adjacency: for each element, which triples it participates in
    triple_index = {i: [] for i in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(forbidden_triples):
        triple_index[a].append(idx)
        triple_index[b].append(idx)
        triple_index[c].append(idx)

    # Conflict graph: two elements conflict if they appear together in some forbidden triple
    conflict = [set() for _ in range(n + 1)]
    for (a, b, c) in forbidden_triples:
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p

    core_set = prime_powers | {1}

    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    remaining = [x for x in range(1, n + 1) if x not in core_set]

    # Multiple randomized greedy passes
    best_set = set(core_set)

    orderings = []
    orderings.append(sorted(remaining, key=lambda x: (num_distinct_primes(x), x)))
    orderings.append(sorted(remaining, key=lambda x: (num_distinct_primes(x), -x)))
    orderings.append(sorted(remaining, key=lambda x: x))
    orderings.append(sorted(remaining, key=lambda x: -x))

    for _ in range(12):
        rand_order = remaining[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)

    def can_add_fast(x, S):
        groups = {}
        for a in S:
            L = get_lcm(x, a)
            if L not in groups:
                groups[L] = []
            groups[L].append(a)

        for L, elems in groups.items():
            if len(elems) < 2:
                continue
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    if get_lcm(elems[i], elems[j]) == L:
                        return False
        return True

    for ordering in orderings:
        if time.time() > deadline - 15.0:
            break

        current = set(core_set)
        for x in ordering:
            if time.time() > deadline - 15.0:
                break
            if x in current:
                continue
            if can_add_fast(x, current):
                current.add(x)

        if len(current) > len(best_set):
            best_set = set(current)

    # Local search: add/remove heuristics
    def count_violations(S):
        """Count number of violated triples in set S."""
        count = 0
        S_set = set(S)
        for (a, b, c) in forbidden_triples:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count

    def local_search(current_set, max_steps=50000):
        """Local search with add/remove heuristics."""
        current = set(current_set)
        best = set(current)
        best_size = len(best)

        # Initial violation count
        cur_viol = count_violations(current)

        steps_since_improve = 0
        max_steps_no_improve = 5000

        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break

            if steps_since_improve > max_steps_no_improve:
                # Random restart: flip 2-3 random elements
                all_nums = list(range(1, n + 1))
                for _ in range(random.randint(2, 3)):
                    x = random.choice(all_nums)
                    if x in current:
                        current.discard(x)
                    else:
                        current.add(x)
                cur_viol = count_violations(current)
                steps_since_improve = 0
                continue

            # Try to add a safe element (one that doesn't create violations)
            safe_adds = []
            for x in range(1, n + 1):
                if x not in current:
                    # Check if adding x creates any new violations
                    creates_violation = False
                    for tidx in triple_index[x]:
                        a, b, c = forbidden_triples[tidx]
                        if x == a:
                            other1, other2 = b, c
                        elif x == b:
                            other1, other2 = a, c
                        else:
                            other1, other2 = a, b
                        if other1 in current and other2 in current:
                            creates_violation = True
                            break
                    if not creates_violation:
                        safe_adds.append(x)

            if safe_adds and random.random() < 0.7:
                x = random.choice(safe_adds)
                current.add(x)
                cur_viol = count_violations(current)
            else:
                # Try to remove an element that fixes violations
                removals = []
                for x in current:
                    fix_count = 0
                    for tidx in triple_index[x]:
                        a, b, c = forbidden_triples[tidx]
                        if a in current and b in current and c in current:
                            fix_count += 1
                    if fix_count > 0:
                        removals.append((x, fix_count))

                if removals:
                    x, f = max(removals, key=lambda t: t[1])
                    current.discard(x)
                    cur_viol = count_violations(current)
                else:
                    steps_since_improve += 1
                    continue

            # Track best
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
                steps_since_improve = 0
            else:
                steps_since_improve += 1

        return best

    # Run local search from current best
    improved = True
    ls_passes = 0
    while improved and time.time() < deadline - 5.0 and ls_passes < 15:
        improved = False
        ls_passes += 1
        ls_result = local_search(best_set, max_steps=30000)
        if len(ls_result) > len(best_set):
            best_set = ls_result
            improved = True

    # Branch and bound with conflict graph coloring upper bound
    def greedy_color_bound(candidates, included):
        """
        Compute an upper bound on the size of an independent set in the conflict graph
        restricted to candidates, using greedy coloring of the subgraph induced by candidates.
        The number of colors used is an upper bound on the independence number.
        """
        if not candidates:
            return 0

        # Build the subgraph on candidates
        cand_set = set(candidates)
        colors = {}
        next_color = 0

        # Order candidates by degree in the subgraph (descending)
        def degree_in_subgraph(x):
            return len([y for y in cand_set if y != x and y in conflict[x]])

        ordered = sorted(candidates, key=degree_in_subgraph, reverse=True)

        for x in ordered:
            # Find the smallest color not used by neighbors in the subgraph
            used_colors = set()
            for y in cand_set:
                if y != x and y in conflict[x]:
                    if y in colors:
                        used_colors.add(colors[y])
            c = 0
            while c in used_colors:
                c += 1
            colors[x] = c
            next_color = max(next_color, c + 1)

        return next_color

    def branch_and_bound(included, candidates, best_size):
        """
        Branch and bound to find a larger independent set.
        included: set of elements already included
        candidates: list of elements to consider adding
        best_size: current best size
        Returns the best set found.
        """
        if time.time() > deadline - 3.0:
            return set(included)

        if not candidates:
            return set(included)

        # Upper bound: current included + maximum additional from candidates
        ub = len(included) + greedy_color_bound(candidates, included)
        if ub <= best_size:
            return set(included)

        # Choose the candidate with the highest degree in the conflict graph
        def degree_in_candidates(x):
            return len([y for y in candidates if y != x and y in conflict[x]])

        ordered = sorted(candidates, key=degree_in_candidates, reverse=True)
        pivot = ordered[0]

        # Branch 1: include pivot
        new_included = set(included)
        new_included.add(pivot)
        # Filter candidates that conflict with pivot
        new_candidates = [x for x in candidates if x != pivot and x not in conflict[pivot]]
        result1 = branch_and_bound(new_included, new_candidates, best_size)

        # Branch 2: exclude pivot
        new_candidates2 = [x for x in candidates if x != pivot]
        result2 = branch_and_bound(included, new_candidates2, best_size)

        # Return the larger set
        if len(result1) >= len(result2):
            return result1
        else:
            return result2

    # Run B&B if we have time
    if time.time() < deadline - 10.0:
        all_nums = list(range(1, n + 1))
        candidates_bb = [x for x in all_nums if x not in best_set]

        if candidates_bb:
            # Limit B&B to a reasonable depth
            # Only consider candidates that don't conflict with the included set
            safe_candidates = [x for x in candidates_bb if not (conflict[x] & best_set)]

            if safe_candidates:
                bb_result = branch_and_bound(set(best_set), safe_candidates, len(best_set))
                if len(bb_result) > len(best_set):
                    best_set = bb_result

    # Final validation: remove any elements causing violations
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        S = sorted(best_set)
        for i in range(len(S)):
            if time.time() > deadline - 1.0:
                break
            for j in range(i + 1, len(S)):
                lab = get_lcm(S[i], S[j])
                for k in range(j + 1, len(S)):
                    if get_lcm(S[i], S[k]) == lab and get_lcm(S[j], S[k]) == lab:
                        # Remove the element in the triple with highest triple_index count
                        counts = [len(triple_index[S[i]]), len(triple_index[S[j]]), len(triple_index[S[k]])]
                        max_idx = counts.index(max(counts))
                        to_remove = S[i] if max_idx == 0 else (S[j] if max_idx == 1 else S[k])
                        best_set.discard(to_remove)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    return list(best_set) if len(best_set) > 0 else list(core_set)
