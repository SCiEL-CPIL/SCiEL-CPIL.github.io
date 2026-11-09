"""Rewritten program: greedy multi-start + local-search for large sum-free subsets."""
import time

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # wall-clock limit; keep margin
    MARGIN = 2.0      # safety margin seconds

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
        """Check if set S is sum-free: no a+b=c for a,b,c in S."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def greedy_build(S, order):
        """Greedy: add elements in given order if sum-free is maintained."""
        chosen = []
        chosen_set = set()
        for idx in order:
            x = S[idx]
            ok = True
            for a in chosen_set:
                if (x - a) in chosen_set:
                    ok = False
                    break
                if a + x in chosen_set:
                    ok = False
                    break
            if ok and 2 * x in chosen_set:
                ok = False
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def local_search_swap(S, initial):
        """Local search with swap moves: remove one element, try to add one (net size same or +1).
        Also try pure additions. Returns improved set."""
        current = list(initial)
        current_set = set(current)
        best = len(current)
        n = len(S)
        
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() - start_time > DEADLINE - MARGIN:
                break
                
            # First, try pure additions (no removal)
            for y in S:
                if y in current_set:
                    continue
                ok = True
                for a in current_set:
                    if (y - a) in current_set:
                        ok = False
                        break
                    if a + y in current_set:
                        ok = False
                        break
                if ok and 2 * y in current_set:
                    ok = False
                if ok:
                    current.append(y)
                    current_set.add(y)
                    best = len(current)
                    improved = True
            
            # Then, try swap moves: remove x, add y
            if not improved:
                for x in current:
                    # Try removing x
                    test_set = set(current_set)
                    test_set.discard(x)
                    for y in S:
                        if y in test_set:
                            continue
                        # Check if test_set ∪ {y} is sum-free
                        ok = True
                        for a in test_set:
                            if (y - a) in test_set:
                                ok = False
                                break
                            if a + y in test_set:
                                ok = False
                                break
                        if ok and 2 * y in test_set:
                            ok = False
                        if ok:
                            # Accept the swap
                            current_set.discard(x)
                            current_set.add(y)
                            current = list(current_set)
                            best = len(current)
                            improved = True
                            break
                    if improved:
                        break
        return current

    def polish_pass(S, initial):
        """Final polish: try to add back any element that fits, in multiple passes."""
        current = list(initial)
        current_set = set(current)
        n = len(S)
        
        for _ in range(3):  # Multiple passes
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            added = False
            for y in S:
                if y in current_set:
                    continue
                ok = True
                for a in current_set:
                    if (y - a) in current_set:
                        ok = False
                        break
                    if a + y in current_set:
                        ok = False
                        break
                if ok and 2 * y in current_set:
                    ok = False
                if ok:
                    current.append(y)
                    current_set.add(y)
                    added = True
            if not added:
                break
        return current

    def solve_instance(S, time_budget):
        """Solve one instance: multiple greedy starts + local search + polish."""
        n = len(S)
        best_result = []
        best_size = 0
        
        # Baseline: odd numbers
        odds = [x for x in S if x % 2 != 0]
        if is_sum_free(odds):
            best_result = odds
            best_size = len(odds)
        
        # Try multiple greedy orderings
        import random
        orderings = []
        
        # Natural order
        orderings.append(list(range(n)))
        # Reverse order
        orderings.append(list(range(n - 1, -1, -1)))
        
        # Random shuffles
        rng = random.Random(42)
        num_random = 30
        for _ in range(num_random):
            perm = list(range(n))
            rng.shuffle(perm)
            orderings.append(perm)
        
        # Sorted by absolute value
        orderings.append(sorted(range(n), key=lambda i: abs(S[i])))
        # Sorted by value
        orderings.append(sorted(range(n), key=lambda i: S[i]))
        # Sorted by -value
        orderings.append(sorted(range(n), key=lambda i: -S[i]))
        # Sorted by value/2 (evenness preference)
        orderings.append(sorted(range(n), key=lambda i: S[i] / 2.0))
        
        # Track all greedy results for later local search
        greedy_results = []
        for order in orderings:
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            result = greedy_build(S, order)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
            greedy_results.append(result)
        
        # Local search on best result and some other promising ones
        candidates = [best_result]
        # Add top few greedy results by size
        greedy_results.sort(key=len, reverse=True)
        for gr in greedy_results[1:4]:
            if gr != best_result and len(gr) >= best_size - 2:
                candidates.append(gr)
        
        for cand in candidates:
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            ls_result = local_search_swap(S, cand)
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best_result = ls_result
        
        # Final polish
        if time.time() - start_time < DEADLINE - MARGIN:
            polished = polish_pass(S, best_result)
            if len(polished) > best_size:
                best_size = len(polished)
                best_result = polished
        
        return best_result

    # Solve all three instances
    instances = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        instances.append(S)
    
    results = []
    for S in instances:
        if time.time() - start_time > DEADLINE - MARGIN:
            # Fallback to odd numbers
            results.append([x for x in S if x % 2 != 0])
        else:
            result = solve_instance(S, DEADLINE - MARGIN - (time.time() - start_time))
            # Verify
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)
    
    return results
