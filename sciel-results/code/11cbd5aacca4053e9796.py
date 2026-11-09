"""Stronger sum-free subset construction with multiple strategies and time budget."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free_fast(S):
        """Faster sum-free check using set operations."""
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def greedy_sum_free(S, order):
        """Greedy: add elements in given order if sum-free is maintained."""
        result = []
        result_set = set()
        for x in order:
            ok = True
            for a in result_set:
                if x - a in result_set:
                    ok = False
                    break
                if x + a in result_set:
                    ok = False
                    break
            if ok:
                result.append(x)
                result_set.add(x)
        return result

    def odd_subset(S):
        """All odd elements are sum-free (odd+odd=even)."""
        return [x for x in S if x % 2 != 0]

    def best_subset_for_instance(S):
        """Try multiple strategies and return the best sum-free subset."""
        candidates = []

        # Strategy 1: Odd numbers
        odd = odd_subset(S)
        if is_sum_free_fast(odd):
            candidates.append(odd)

        # Strategy 2: Greedy by abs desc
        order2 = sorted(S, key=lambda x: -abs(x))
        g2 = greedy_sum_free(S, order2)
        if is_sum_free_fast(g2):
            candidates.append(g2)

        # Strategy 3: Greedy by abs asc
        order3 = sorted(S, key=lambda x: abs(x))
        g3 = greedy_sum_free(S, order3)
        if is_sum_free_fast(g3):
            candidates.append(g3)

        # Strategy 4: Greedy by value desc
        order4 = sorted(S, reverse=True)
        g4 = greedy_sum_free(S, order4)
        if is_sum_free_fast(g4):
            candidates.append(g4)

        # Strategy 5: Greedy by value asc
        order5 = sorted(S)
        g5 = greedy_sum_free(S, order5)
        if is_sum_free_fast(g5):
            candidates.append(g5)

        # Find best candidate
        best = []
        for c in candidates:
            if len(c) > len(best):
                best = c

        # If best is empty, fall back to odd
        if not best:
            best = odd_subset(S)

        return best

    instances_params = [(60, 801), (120, 802), (240, 803)]
    results = []

    for size, seed in instances_params:
        S = instance(size, seed)
        subset = best_subset_for_instance(S)
        results.append(subset)

        # Check time
        if time.time() > DEADLINE - 10:
            break

    # Fill remaining with odd subsets if needed
    for i in range(len(results), 3):
        size, seed = instances_params[i]
        S = instance(size, seed)
        results.append(odd_subset(S))

    # Verify all are sum-free, fallback to odd if not
    final_results = []
    for i, (size, seed) in enumerate(instances_params):
        if i < len(results) and is_sum_free_fast(results[i]):
            final_results.append(results[i])
        else:
            S = instance(size, seed)
            final_results.append(odd_subset(S))

    return final_results
# EVOLVE-BLOCK-END
