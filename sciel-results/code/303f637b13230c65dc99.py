"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 115  # 15s safety margin

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
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_sum_free(S, start_time=None):
        """Greedy: sort by |x|, add x if sum-free with current set."""
        if start_time is None:
            start_time = time.time()
        S_sorted = sorted(S, key=lambda x: (abs(x), x))
        result = []
        result_set = set()
        for x in S_sorted:
            # Check if adding x violates sum-freeness
            # Need: no a, b in result_set such that a + b = x
            # And no a in result_set such that a + x in result_set (but that's covered by a+x = c where c in result_set)
            # Actually: for all a, b in result_set, a + b != x
            # And for all a in result_set, a + x not in result_set
            # And x + x not in result_set
            ok = True
            for a in result_set:
                if x - a in result_set and a <= x - a:
                    # a + (x-a) = x, both in result_set
                    ok = False
                    break
                if a + x in result_set:
                    ok = False
                    break
            if ok and 2 * x not in result_set:
                result.append(x)
                result_set.add(x)
        return result

    def local_search_improve(S, initial, start_time=None):
        """Try to improve initial sum-free subset by adding/removing elements."""
        if start_time is None:
            start_time = time.time()
        current = list(initial)
        current_set = set(current)
        S_set = set(S)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try adding each element not in current
            for x in S:
                if x in current_set:
                    continue
                # Check if we can add x
                ok = True
                for a in current_set:
                    if x - a in current_set and a <= x - a:
                        ok = False
                        break
                    if a + x in current_set:
                        ok = False
                        break
                if ok and 2 * x not in current_set:
                    current.append(x)
                    current_set.add(x)
                    improved = True
            # Try removing one element and adding two
            if time.time() > deadline:
                break
            for i, a in enumerate(current):
                if time.time() > deadline:
                    break
                temp_set = set(current)
                temp_set.remove(a)
                # Try adding elements from S not in temp_set
                added = 0
                new_elements = []
                for x in S:
                    if x in temp_set:
                        continue
                    ok = True
                    for b in temp_set:
                        if x - b in temp_set and b <= x - b:
                            ok = False
                            break
                        if b + x in temp_set:
                            ok = False
                            break
                    if ok and 2 * x not in temp_set:
                        temp_set.add(x)
                        new_elements.append(x)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    # This move improves size
                    current = list(temp_set)
                    current_set = temp_set
                    improved = True

        return current

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Start with odd numbers
        odd_subset = [x for x in S if x % 2]
        
        # Also try greedy
        greedy_subset = greedy_sum_free(S, start_time)
        
        # Pick the better starting point
        if len(odd_subset) >= len(greedy_subset):
            best = odd_subset
        else:
            best = greedy_subset
        
        # Try to improve with local search
        if time.time() < deadline:
            improved = local_search_improve(S, best, start_time)
            if len(improved) > len(best):
                best = improved
        
        # Verify sum-freeness
        if not is_sum_free(best):
            # Fallback to odd numbers which are guaranteed sum-free
            best = odd_subset
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END
