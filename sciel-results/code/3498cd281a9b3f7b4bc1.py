import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute all forbidden triples (a,b,c) with a<b<c, 1/a = 1/b + 1/c
    # Equivalent to: bc = a(b+c) => c = ab/(b-a)
    triples = []
    triple_map = defaultdict(set)

    for a in range(1, 181):
        for b in range(a + 1, 181):
            den = b - a
            if den == 0:
                continue
            num = a * b
            if num % den == 0:
                c = num // den
                if c > b and c <= 180:
                    triple = (a, b, c)
                    triples.append(triple)
                    triple_map[a].add(triple)
                    triple_map[b].add(triple)
                    triple_map[c].add(triple)

    # For each element, precompute the pairs that would conflict
    conflict_pairs = defaultdict(list)
    for (a, b, c) in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                return True
        return False

    def greedy_build(order, S_init=None):
        """Build a set greedily following the given order."""
        S = set(S_init) if S_init is not None else set()
        for x in order:
            if x in S:
                continue
            if not check_conflict(x, S):
                S.add(x)
        return S

    best_S = set()
    best_size = 0

    # Strategy 1: Start with large numbers, add small ones
    S1 = set(range(91, 181))
    candidates = list(range(1, 91))
    random.shuffle(candidates)
    for x in candidates:
        if not check_conflict(x, S1):
            S1.add(x)
    if len(S1) > best_size:
        best_size = len(S1)
        best_S = set(S1)

    # Strategy 2: Pure greedy from all numbers in random order
    all_nums = list(range(1, 181))
    random.shuffle(all_nums)
    S2 = greedy_build(all_nums)
    if len(S2) > best_size:
        best_size = len(S2)
        best_S = set(S2)

    # Strategy 3: Greedy with small numbers first
    S3 = greedy_build(list(range(1, 181)))
    if len(S3) > best_size:
        best_size = len(S3)
        best_S = set(S3)

    # Strategy 4: Greedy with large numbers first
    S4 = greedy_build(list(range(180, 0, -1)))
    if len(S4) > best_size:
        best_size = len(S4)
        best_S = set(S4)

    # Strategy 5: Multiple random restarts
    for _ in range(15):
        if time.time() > deadline - 50:
            break
        nums = list(range(1, 181))
        random.shuffle(nums)
        S5 = greedy_build(nums)
        if len(S5) > best_size:
            best_size = len(S5)
            best_S = set(S5)

    # Strategy 6: Weighted greedy - prioritize elements with fewer conflicts
    if time.time() < deadline - 45:
        degree = {x: len(conflict_pairs[x]) for x in range(1, 181)}
        order = sorted(range(1, 181), key=lambda x: degree[x])
        S6 = greedy_build(order)
        if len(S6) > best_size:
            best_size = len(S6)
            best_S = set(S6)

    # Local search on best_S
    S = set(best_S)

    # Phase 1: Try to add any remaining elements
    remaining = [x for x in range(1, 181) if x not in S]
    for x in remaining:
        if not check_conflict(x, S):
            S.add(x)
    best_size = len(S)
    best_S = set(S)

    # Phase 2: Local search - remove one, add multiple
    while time.time() < deadline - 20:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        remaining = [x for x in range(1, 181) if x not in S]

        for remove_elem in S_list:
            if time.time() > deadline - 20:
                break
            S.remove(remove_elem)

            added = []
            for x in remaining:
                if x in S:
                    continue
                if not check_conflict(x, S):
                    S.add(x)
                    added.append(x)

            if len(added) >= 1 and len(S) > len(best_S):
                best_size = len(S)
                best_S = set(S)
                improved = True
                remaining = [x for x in range(1, 181) if x not in S]
            else:
                for x in added:
                    S.discard(x)
                S.add(remove_elem)

        if not improved:
            break

    # Phase 3: Branch and bound with matching-based lower bound
    if time.time() < deadline - 5:
        def compute_matching_bound(rem_set, active_triples):
            """Greedy maximum matching on hypergraph for lower bound on exclusions."""
            matched = set()
            count = 0
            sorted_triples = sorted(active_triples, key=lambda t: sum(1 for x in t if x not in matched and x in rem_set), reverse=True)
            for t in sorted_triples:
                unmatched_in_t = [x for x in t if x not in matched and x in rem_set]
                if not unmatched_in_t:
                    continue
                count += 1
                for x in unmatched_in_t:
                    matched.add(x)
                if len(matched) >= len(rem_set):
                    break
            return count

        def bb_search(rem, inc, exc, depth):
            nonlocal best_size, best_S
            if time.time() > deadline - 3 or depth < 0:
                return

            if not rem:
                if len(inc) > best_size:
                    best_size = len(inc)
                    best_S = set(inc)
                return

            ub = len(inc) + len(rem)
            if ub <= best_size:
                return

            active = [t for t in triples if any(x in rem for x in t) and not any(x in exc for x in t)]
            match_lb = compute_matching_bound(rem, active)
            ub2 = len(inc) + len(rem) - match_lb
            if ub2 <= best_size:
                return

            elem_counts = defaultdict(int)
            for t in active:
                for x in t:
                    if x in rem:
                        elem_counts[x] += 1

            if not elem_counts:
                new_inc = inc | rem
                if len(new_inc) > best_size:
                    best_size = len(new_inc)
                    best_S = set(new_inc)
                return

            branch_e = max(elem_counts, key=elem_counts.get)

            bb_search(rem - {branch_e}, inc, exc | {branch_e}, depth - 1)
            bb_search(rem - {branch_e}, inc | {branch_e}, exc, depth - 1)

        rem_init = set(range(1, 181)) - set(best_S)
        inc_init = set(best_S)
        exc_init = set()
        bb_search(rem_init, inc_init, exc_init, 15)

    result = list(best_S)
    if not result:
        result = list(range(91, 181))

    return result
