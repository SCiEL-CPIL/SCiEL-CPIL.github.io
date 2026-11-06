import time
import math
import random
import sys
import heapq

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if d == 0:
                continue
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    N = 180

    triples = _precompute_triples(N)
    T = len(triples)

    # Build index: element -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    # Precompute degree of each element
    elem_degree = {i: len(elem_triples[i]) for i in range(1, N + 1)}

    best_set = set()
    best_size = 0

    def greedy_removal(random_seed, strategy):
        """
        Strategy 0: Remove element with max active triple count (ties broken randomly).
        Strategy 1: Remove element with max active triple count, prefer smaller numbers.
        Strategy 2: Remove element with max (active_count * 10 + degree) as tiebreaker.
        """
        random.seed(random_seed)
        S = set(range(1, N + 1))
        counts = [3] * T  # Initially all in S

        active_count = [0] * (N + 1)
        for t_idx in range(T):
            a, b, c = triples[t_idx]
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1

        # Use a heap for efficient max extraction: (-score, element)
        # Score depends on strategy
        def get_score(x):
            if strategy == 0:
                return active_count[x]
            elif strategy == 1:
                # Prefer smaller numbers on ties
                return (active_count[x], -x)
            else:
                # Weight by degree
                return active_count[x] * 10 + elem_degree[x]

        # We'll use a simpler approach: rebuild candidate list each iteration
        # Since T is small (~300), this is fast enough
        while True:
            if time.time() > deadline - 5.0:
                return S

            # Find active triples
            has_active = False
            max_score = -1
            x_to_remove = None
            best_score = None

            for x in S:
                ac = active_count[x]
                if ac <= 0:
                    continue
                has_active = True
                if strategy == 0:
                    score = ac
                elif strategy == 1:
                    score = (ac, -x)
                else:
                    score = ac * 10 + elem_degree[x]

                if best_score is None or score > best_score:
                    best_score = score
                    x_to_remove = x
                    max_score = ac

            if not has_active or x_to_remove is None:
                return S

            S.remove(x_to_remove)

            # Update counts and active_count for triples containing x_to_remove
            for t_idx in elem_triples[x_to_remove]:
                if counts[t_idx] == 3:
                    a, b, c = triples[t_idx]
                    active_count[a] -= 1
                    active_count[b] -= 1
                    active_count[c] -= 1
                    counts[t_idx] = 2
                elif counts[t_idx] == 2:
                    counts[t_idx] = 1

        return S

    def local_search_add(S, random_seed):
        """
        Try to add back removed elements.
        """
        random.seed(random_seed)
        S = set(S)

        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1

        removed = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(removed)

        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            for x in removed:
                if x in S:
                    continue
                can_add = True
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    improved = True
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1

        return S

    def swap_local_search(S, random_seed):
        """
        Try 2-opt moves: remove one element x from S, then add as many removed elements as possible.
        If we can add more than 1, accept the move (net gain > 0).
        """
        random.seed(random_seed)
        S = set(S)

        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1

        removed = [x for x in range(1, N + 1) if x not in S]

        S_list = list(S)
        random.shuffle(S_list)

        # Limit number of swap attempts based on time
        max_swaps = 80
        for x in S_list[:max_swaps]:
            if time.time() > deadline - 5.0:
                break

            # Temporarily remove x
            S.remove(x)
            for t_idx in elem_triples[x]:
                counts[t_idx] -= 1

            # Try to add as many removed elements as possible
            added = []
            for y in removed:
                if y in S:
                    continue
                can_add = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(y)
                    added.append(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1

            # If we added more than 1, accept (net gain)
            if len(added) <= 1:
                # Revert: remove added elements, put x back
                for y in added:
                    S.remove(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] -= 1
                S.add(x)
                for t_idx in elem_triples[x]:
                    counts[t_idx] += 1

        # Final pass: try to add any remaining removable elements
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            for y in removed:
                if y in S:
                    continue
                can_add = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(y)
                    improved = True
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1

        return S

    def triple_swap_search(S, random_seed):
        """
        More aggressive: try removing up to 2 elements and adding back more.
        """
        random.seed(random_seed)
        S = set(S)

        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1

        removed = [x for x in range(1, N + 1) if x not in S]
        S_list = list(S)
        random.shuffle(S_list)

        # Try pairs of removals
        max_pairs = 40
        for i in range(min(max_pairs, len(S_list) - 1)):
            if time.time() > deadline - 5.0:
                break
            x1 = S_list[i]
            x2 = S_list[i + 1]

            # Temporarily remove x1 and x2
            S.remove(x1)
            S.remove(x2)
            for t_idx in elem_triples[x1]:
                counts[t_idx] -= 1
            for t_idx in elem_triples[x2]:
                counts[t_idx] -= 1

            # Try to add as many removed elements as possible
            added = []
            for y in removed:
                if y in S:
                    continue
                can_add = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(y)
                    added.append(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1

            # If we added more than 2, accept (net gain)
            if len(added) <= 2:
                # Revert
                for y in added:
                    S.remove(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] -= 1
                S.add(x1)
                S.add(x2)
                for t_idx in elem_triples[x1]:
                    counts[t_idx] += 1
                for t_idx in elem_triples[x2]:
                    counts[t_idx] += 1

        # Final pass
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            for y in removed:
                if y in S:
                    continue
                can_add = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(y)
                    improved = True
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1

        return S

    # Run multiple restarts with different strategies
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline - 3.0:
            break

        strategy = i % 3
        S_greedy = greedy_removal(i, strategy)
        S_local = local_search_add(S_greedy, i + 1000)
        S_swap = swap_local_search(S_local, i + 2000)
        S_final = triple_swap_search(S_swap, i + 3000)

        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)

    # If we have time, do a few more rounds of polishing on the best set
    if time.time() < deadline - 10.0 and best_size > 0:
        for polish in range(5):
            if time.time() > deadline - 5.0:
                break
            S_polished = swap_local_search(best_set, 10000 + polish * 100)
            if len(S_polished) > best_size:
                best_size = len(S_polished)
                best_set = set(S_polished)

    # Final validity check
    if not best_set:
        best_set = set(range(90, 181))

    valid = True
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break

    if not valid:
        # Fix by removing elements from conflicting triples
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                # Remove the element with highest degree to minimize impact
                deg_a = elem_degree.get(a, 0)
                deg_b = elem_degree.get(b, 0)
                deg_c = elem_degree.get(c, 0)
                if deg_a >= deg_b and deg_a >= deg_c:
                    best_set.remove(a)
                elif deg_b >= deg_a and deg_b >= deg_c:
                    best_set.remove(b)
                else:
                    best_set.remove(c)

    return list(best_set)

# EVOLVE-BLOCK-END
