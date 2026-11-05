import math
import time
import random

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit

    # Prime factors of 255
    primes = [3, 5, 17]

    # Base best set: all multiples of 3 (size 85)
    best_set = list(range(3, n + 1, 3))
    max_size = len(best_set)

    # Candidates: all numbers in [1,255] divisible by 3, 5, or 17
    candidates = set()
    for p in primes:
        for x in range(p, n + 1, p):
            candidates.add(x)
    candidates = sorted(candidates)

    m = len(candidates)

    # Build compatibility matrix
    # Two candidates are compatible if gcd > 1
    compat = [[False] * m for _ in range(m)]
    for i in range(m):
        compat[i][i] = True
        for j in range(i + 1, m):
            if math.gcd(candidates[i], candidates[j]) > 1:
                compat[i][j] = True
                compat[j][i] = True

    # Find index of 255
    idx_255 = candidates.index(255)

    # Verify the base clique: all multiples of 3 are pairwise compatible
    # (they all share factor 3)
    base_clique_indices = [candidates.index(x) for x in best_set]

    # Try to expand the base clique by adding elements from B\A or C\A
    # that are compatible with ALL elements in the base clique
    def can_add(idx, clique_indices, compat, m):
        for j in clique_indices:
            if not compat[idx][j]:
                return False
        return True

    # Try adding elements not in the base clique
    base_set_indices = set(base_clique_indices)
    expanded_clique = list(base_clique_indices)

    for i in range(m):
        if i in base_set_indices:
            continue
        if can_add(i, expanded_clique, compat, m):
            expanded_clique.append(i)

    if len(expanded_clique) > max_size:
        best_set = [candidates[i] for i in expanded_clique]
        max_size = len(best_set)

    # Try greedy clique finding from 255 with different orderings
    def greedy_clique(containing_idx, compat, m, order=None, time_limit=5.0):
        start_t = time.time()
        current = {containing_idx}

        if order is None:
            # Default: sort by degree descending
            degrees = [sum(1 for j in range(m) if compat[i][j]) for i in range(m)]
            order = sorted(range(m), key=lambda x: -degrees[x])

        for i in order:
            if time.time() - start_t > time_limit:
                break
            if i in current:
                continue
            if not compat[i][containing_idx]:
                continue
            # Check if i is compatible with all in current
            ok = True
            for j in current:
                if not compat[i][j]:
                    ok = False
                    break
            if ok:
                current.add(i)

        return current

    # Try greedy with degree ordering
    degrees = [sum(1 for j in range(m) if compat[i][j]) for i in range(m)]
    degree_order = sorted(range(m), key=lambda x: -degrees[x])
    greedy_result = greedy_clique(idx_255, compat, m, order=degree_order, time_limit=5.0)
    if len(greedy_result) > max_size:
        best_set = [candidates[i] for i in greedy_result]
        max_size = len(greedy_result)

    # Try random restarts with time budget
    if time.time() < deadline - 15:
        random.seed(42)
        for restart in range(200):
            if time.time() > deadline - 10:
                break

            current = {idx_255}
            # Get all candidates compatible with 255
            compatible_with_255 = [i for i in range(m) if i != idx_255 and compat[i][idx_255]]
            random.shuffle(compatible_with_255)

            for i in compatible_with_255:
                if time.time() > deadline - 10:
                    break
                ok = True
                for j in current:
                    if not compat[i][j]:
                        ok = False
                        break
                if ok:
                    current.add(i)

            if len(current) > max_size:
                best_set = [candidates[i] for i in current]
                max_size = len(current)

    # Try local search: start from best_set, try to swap in elements
    if time.time() < deadline - 10:
        best_clique_set = set(best_set)
        best_clique_indices = set(candidates.index(x) for x in best_set)

        # Try to add any candidate not in current set
        for i in range(m):
            if i in best_clique_indices:
                continue
            if can_add(i, list(best_clique_indices), compat, m):
                best_clique_indices.add(i)

        if len(best_clique_indices) > max_size:
            best_set = [candidates[i] for i in best_clique_indices]
            max_size = len(best_set)

    # Final verification
    assert 255 in best_set
    assert len(best_set) > 0
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            assert math.gcd(best_set[i], best_set[j]) > 1, f"Pair ({best_set[i]}, {best_set[j]}) has gcd 1"

    return best_set
