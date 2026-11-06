import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-free property."""
        # For each a in S, check a+x in S and x-a in S
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_add(S_list, candidates, S_set=None):
        """Greedily add candidates to S_list if possible."""
        if S_set is None:
            S_set = set(S_list)
        result = list(S_list)
        for c in candidates:
            if c in S_set:
                continue
            if can_add(c, S_set):
                result.append(c)
                S_set.add(c)
        return result, S_set

    def local_search(S_list, values, time_limit=5.0):
        """Improve S_list using add, remove-and-add, and swap moves."""
        S_set = set(S_list)
        values_set = set(values)
        start = time.time()
        S_list = list(S_list)
        
        # Phase 1: Try adding any missing element
        missing = list(values_set - S_set)
        if missing:
            S_list, S_set = greedy_add(S_list, missing, S_set)
        
        # Phase 2: Iterative improvement - try removing one element and adding others
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            S_copy = list(S_list)
            S_set_copy = set(S_set)
            
            for x in S_copy:
                if (time.time() - start) >= time_limit:
                    break
                
                # Remove x
                S_test = S_set_copy - {x}
                S_test_list = [y for y in S_copy if y != x]
                
                # Try to add all missing elements greedily
                cand_list = [y for y in values if y not in S_test]
                S_new_list, S_new_set = greedy_add(S_test_list, cand_list, S_test)
                
                if len(S_new_list) > len(S_list):
                    S_list = S_new_list
                    S_set = S_new_set
                    improved = True
                    break
        
        # Phase 3: Try swap - remove one, add two specific elements
        if (time.time() - start) < time_limit:
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                S_copy = list(S_list)
                S_set_copy = set(S_set)
                
                for x in S_copy:
                    if (time.time() - start) >= time_limit:
                        break
                    
                    # Remove x
                    S_test = S_set_copy - {x}
                    S_test_list = [y for y in S_copy if y != x]
                    
                    # Try adding pairs of missing elements
                    missing = list(values_set - S_test)
                    added = 0
                    for i, a in enumerate(missing):
                        if (time.time() - start) >= time_limit:
                            break
                        if a in S_test:
                            continue
                        if can_add(a, S_test):
                            S_test.add(a)
                            S_test_list.append(a)
                            added += 1
                            if added >= 2:
                                break
                    
                    if len(S_test_list) > len(S_list):
                        S_list = S_test_list
                        S_set = S_test
                        improved = True
                        break
        
        # Phase 4: 2-opt swaps - remove two, add up to three
        if (time.time() - start) < time_limit:
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                S_copy = list(S_list)
                n = len(S_copy)
                
                for i in range(n):
                    if (time.time() - start) >= time_limit:
                        break
                    for j in range(i+1, n):
                        if (time.time() - start) >= time_limit:
                            break
                        
                        x1, x2 = S_copy[i], S_copy[j]
                        if x1 == x2:
                            continue
                        
                        # Remove both
                        S_test = S_set - {x1, x2}
                        S_test_list = [y for y in S_copy if y not in (x1, x2)]
                        
                        # Try adding up to 3 missing elements greedily
                        missing = list(values_set - S_test)
                        added = 0
                        for a in missing:
                            if (time.time() - start) >= time_limit:
                                break
                            if can_add(a, S_test):
                                S_test.add(a)
                                S_test_list.append(a)
                                added += 1
                                if added >= 3:
                                    break
                        
                        if len(S_test_list) > len(S_list):
                            S_list = S_test_list
                            S_set = S_test
                            improved = True
                            break
                
                if improved:
                    break
        
        return S_list

    def solve_one(values, time_budget=15.0):
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        V_set = set(values)
        max_v = max(values)
        min_v = min(values)
        
        best = []
        best_set = set()
        
        # Generate multiple initial solutions
        candidates = []
        
        # Strategy 1: Odd numbers
        s1 = [x for x in values if x % 2 != 0]
        candidates.append(s1)
        
        # Strategy 2: Large positive numbers (above max/2)
        thresh_pos = max_v / 2.0
        s2 = [x for x in values if x > thresh_pos]
        candidates.append(s2)
        
        # Strategy 3: Large negative numbers (below min/2)
        thresh_neg = min_v / 2.0
        s3 = [x for x in values if x < thresh_neg]
        candidates.append(s3)
        
        # Strategy 4: Union of large pos and neg
        s4 = sorted(set(s2).union(s3))
        candidates.append(s4)
        
        # Strategy 5: Greedy by absolute value (largest first)
        s5_list, s5_set = greedy_add([], sorted(values, key=abs, reverse=True))
        candidates.append(s5_list)
        
        # Strategy 6: Greedy by value (smallest first)
        s6_list, s6_set = greedy_add([], sorted(values))
        candidates.append(s6_list)
        
        # Strategy 7: Greedy by value (largest first)
        s7_list, s7_set = greedy_add([], sorted(values, reverse=True))
        candidates.append(s7_list)
        
        # Strategy 8: More random shuffles
        rng = Random(42)
        for trial in range(5):
            shuffled = list(values)
            rng.shuffle(shuffled)
            sl, ss = greedy_add([], shuffled)
            candidates.append(sl)
        
        # Strategy 9: Greedy with random perturbation
        for trial in range(3):
            shuffled = list(values)
            rng.shuffle(shuffled)
            # Start with a random subset, then greedily add
            initial_size = max(1, n // 4)
            initial = shuffled[:initial_size]
            # Repair initial to be sum-free
            initial_valid = []
            initial_set = set()
            for x in initial:
                if can_add(x, initial_set):
                    initial_valid.append(x)
                    initial_set.add(x)
            sl, ss = greedy_add(initial_valid, shuffled[initial_size:], initial_set)
            candidates.append(sl)
        
        # Evaluate and keep best
        for c in candidates:
            if not is_sum_free(c):
                # Repair: remove conflicting elements
                c_valid = list(c)
                c_set = set(c_valid)
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(c_valid):
                        x = c_valid[i]
                        conflict = False
                        for y in c_set:
                            if y == x:
                                continue
                            if (x + y) in c_set:
                                conflict = True
                                break
                            if (x - y) in c_set:
                                conflict = True
                                break
                        if conflict:
                            c_valid.pop(i)
                            c_set.remove(x)
                            changed = True
                        else:
                            i += 1
                c = c_valid
            
            if len(c) > len(best):
                best = c
                best_set = set(c)
        
        # Local search on best
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            ts = min(time_budget, remaining_time - 3)
            improved = local_search(best, values, time_limit=ts)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
        
        # Try additional random restarts if time permits
        if time.time() < deadline - 10:
            for r in range(15):
                if time.time() > deadline - 5:
                    break
                shuffled = list(values)
                rng.shuffle(shuffled)
                sl, ss = greedy_add([], shuffled)
                # Quick local search
                ts = min(2.5, deadline - time.time() - 2)
                if ts > 0:
                    improved = local_search(sl, values, time_limit=ts)
                    if len(improved) > len(best):
                        best = improved
                        best_set = set(best)
        
        return list(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Fallback if time is tight
        if time.time() > deadline - 15:
             results.append([x for x in S if x % 2])
             continue
             
        res = solve_one(S)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to parity
            res = [x for x in S if x % 2]
        
        results.append(res)
        
    return results
