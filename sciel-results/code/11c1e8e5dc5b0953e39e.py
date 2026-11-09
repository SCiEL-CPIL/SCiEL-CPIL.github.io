"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin for 150s limit

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) <= 1:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-freeness."""
        # Need: no a,b in S_set with a+b=x, no a in S_set with a+x in S_set, 2x not in S_set
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def greedy_add(S, order_key, start_time=None):
        """Greedy: add elements in given order if sum-free."""
        if start_time is None:
            start_time = time.time()
        S_sorted = sorted(S, key=order_key)
        result = []
        result_set = set()
        for x in S_sorted:
            if time.time() > deadline:
                break
            if can_add(x, result_set):
                result.append(x)
                result_set.add(x)
        return result

    def local_search(S, initial, start_time=None):
        """Fast local search: Greedy add, then limited swaps."""
        if start_time is None:
            start_time = time.time()
        current_set = set(initial)
        S_set = set(S)
        
        # Phase 1: Greedy add
        # Sort candidates by value or random order? Let's use sorted for determinism and speed.
        candidates = sorted([x for x in S if x not in current_set])
        for x in candidates:
            if time.time() > deadline - 5: # Stop if close to deadline
                break
            if can_add(x, current_set):
                current_set.add(x)
        
        # Phase 2: Simple Swap Search (1 remove, 1 add)
        # We only do this if we have time and the set is not too large to iterate
        if time.time() < deadline - 10:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                current_list = list(current_set)
                # Try removing one element and adding one from S
                for a in current_list:
                    if time.time() > deadline - 5:
                        break
                    temp_set = set(current_set)
                    temp_set.discard(a)
                    # Find a candidate to add
                    for x in S:
                        if x in temp_set:
                            continue
                        if can_add(x, temp_set):
                            temp_set.add(x)
                            if len(temp_set) > len(current_set):
                                current_set = temp_set
                                improved = True
                            break
                    if improved:
                        break
        return list(current_set)

    def try_multiple_strategies(S, start_time=None):
        """Try multiple starting strategies and return the best result."""
        if start_time is None:
            start_time = time.time()

        candidates = []

        # Strategy 1: Odd numbers (sum of two odds is even, so sum-free)
        odd_subset = [x for x in S if x % 2 != 0]
        candidates.append(odd_subset)

        # Strategy 2: Greedy by large |x| first
        greedy_large = greedy_add(S, order_key=lambda x: (-abs(x), x), start_time=start_time)
        candidates.append(greedy_large)

        # Strategy 3: Greedy by small |x| first
        greedy_small = greedy_add(S, order_key=lambda x: (abs(x), x), start_time=start_time)
        candidates.append(greedy_small)

        # Strategy 4: Greedy by value (positive first)
        greedy_pos = greedy_add(S, order_key=lambda x: (-x, abs(x)), start_time=start_time)
        candidates.append(greedy_pos)

        # Strategy 5: Greedy by negative value first
        greedy_neg = greedy_add(S, order_key=lambda x: (x, abs(x)), start_time=start_time)
        candidates.append(greedy_neg)

        # Local search on each candidate
        best = None
        best_size = 0
        for cand in candidates:
            if time.time() > deadline:
                break
            if best is None or len(cand) > best_size:
                improved = local_search(S, cand, start_time=start_time)
                if improved is not None and len(improved) > best_size:
                    best = improved
                    best_size = len(improved)
                elif len(cand) > best_size:
                    best = cand
                    best_size = len(cand)
            else:
                # Even if not the best starting point, local search might help
                if time.time() < deadline - 20:  # only if we have time
                    improved = local_search(S, cand, start_time=start_time)
                    if improved is not None and len(improved) > best_size:
                        best = improved
                        best_size = len(improved)

        if best is None:
            best = odd_subset

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        best = try_multiple_strategies(S, start_time)

        # Final verification
        if not is_sum_free(best):
            # Fallback to odd numbers
            best = [x for x in S if x % 2 != 0]

        results.append(best)

    return results
# EVOLVE-BLOCK-END
