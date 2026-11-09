"""Rewritten program: greedy multi-start + local-search for large sum-free subsets."""
import time

def construct():
    start_time = time.time()
    DEADLINE = 150.0  # wall-clock limit; keep margin
    MARGIN = 5.0      # safety margin seconds

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

    def conflict_count(S, idx):
        """Count how many pairs (a,b) in S (excluding idx) have a+b == S[idx]."""
        x = S[idx]
        Sset = set(S)
        Sset.discard(x)
        cnt = 0
        for a in Sset:
            if (x - a) in Sset and a != (x - a):
                cnt += 1
            elif a == (x - a):
                # a + a = x, need a in Sset (it is) and a != x (true since x-a=a => x=2a, a != x unless x=0)
                cnt += 1
        return cnt

    def greedy_build(S, order):
        """Greedy: add elements in given order if sum-free is maintained."""
        chosen = []
        chosen_set = set()
        for idx in order:
            x = S[idx]
            ok = True
            # Check: does adding x create a+b=c?
            # For each a in chosen_set: if a+x in chosen_set or a+x == x (i.e., a=0, but x!=0 and a!=0)
            # Actually check: for all a,b in chosen_set ∪ {x}, a+b != c in chosen_set ∪ {x}
            # Efficient: check if x is "safe" to add
            # Violations involving x:
            # 1) a + b = x for a,b in chosen_set
            # 2) a + x = b for a,b in chosen_set (i.e., b-a = x)
            # 3) x + x = a for a in chosen_set (i.e., 2x in chosen_set)
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

    def local_search(S, initial):
        """Local search: try removing elements to allow adding others; also try swaps."""
        current = list(initial)
        current_set = set(current)
        best = len(current)
        n = len(S)
        Sset = set(S)
        
        # Map value to index in S for quick lookup
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        improved = True
        while improved:
            improved = False
            # Try removing one element
            for i in range(len(current)):
                x = current[i]
                new_set = set(current)
                new_set.discard(x)
                # Try adding any element not in new_set
                for y in S:
                    if y in new_set:
                        continue
                    # Check if new_set ∪ {y} is sum-free
                    ok = True
                    for a in new_set:
                        if (y - a) in new_set:
                            ok = False
                            break
                        if a + y in new_set:
                            ok = False
                            break
                    if ok and 2 * y in new_set:
                        ok = False
                    if ok:
                        # Check if this improves size
                        if len(new_set) + 1 > best:
                            best = len(new_set) + 1
                            current = list(new_set)
                            current = current + [y]
                            current_set = set(current)
                            improved = True
                            break
                if improved:
                    break
        return current

    def solve_instance(S, time_budget):
        """Solve one instance: multiple greedy starts + local search."""
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
        for _ in range(20):
            perm = list(range(n))
            rng.shuffle(perm)
            orderings.append(perm)
        
        # Sorted by absolute value
        orderings.append(sorted(range(n), key=lambda i: abs(S[i])))
        # Sorted by value
        orderings.append(sorted(range(n), key=lambda i: S[i]))
        # Sorted by -value
        orderings.append(sorted(range(n), key=lambda i: -S[i]))
        
        for order in orderings:
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            result = greedy_build(S, order)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
        
        # Local search on best result
        if time.time() - start_time < DEADLINE - MARGIN:
            ls_result = local_search(S, best_result)
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best_result = ls_result
        
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
