import time
import random

def construct():
    start_time = time.time()
    deadline = 145  # 5 seconds margin
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(C_set):
        """Check if set is sum-free."""
        for a in C_set:
            for b in C_set:
                if a + b in C_set:
                    return False
        return True
    
    def count_conflicts(C_set):
        """Count number of sum violations."""
        count = 0
        for a in C_set:
            for b in C_set:
                if a + b in C_set:
                    count += 1
        return count
    
    def get_conflict_elements(C_set):
        """Return dict of element -> number of conflicts it's involved in."""
        conflicts = {x: 0 for x in C_set}
        for a in C_set:
            for b in C_set:
                s = a + b
                if s in C_set:
                    conflicts[a] += 1
                    conflicts[b] += 1
                    conflicts[s] += 1
        return conflicts
    
    def greedy_remove_weighted(S, weight_func, deadline):
        """
        Greedy removal using weighted conflict scoring.
        weight_func: function that gives weight to each element for tie-breaking.
        """
        C = list(S)
        C_set = set(C)
        
        while True:
            if time.time() > deadline:
                break
                
            conflicts = get_conflict_elements(C_set)
            if max(conflicts.values()) == 0:
                break
            
            max_conf = max(conflicts.values())
            candidates = [x for x, c in conflicts.items() if c == max_conf]
            
            # Remove element with highest weight among those with max conflicts
            to_remove = max(candidates, key=weight_func)
            
            C.remove(to_remove)
            C_set.remove(to_remove)
            
        return C
    
    def greedy_add(S, order_key, deadline):
        """Greedy addition in given order."""
        C = []
        C_set = set()
        for x in sorted(S, key=order_key):
            if time.time() > deadline:
                break
            valid = True
            # Check if x creates a violation
            # x + c in C for any c in C
            for c in C:
                if x + c in C_set:
                    valid = False
                    break
            if valid:
                # c - x in C for any c in C (c = x + d)
                for c in C:
                    if c - x in C_set:
                        valid = False
                        break
            if valid:
                # 2x in C
                if 2*x in C_set:
                    valid = False
            
            if valid:
                C.append(x)
                C_set.add(x)
        return C
    
    def local_search_swap(S, initial_set, deadline):
        """
        Try to improve by swapping elements: remove one from set, add one from outside.
        Only accept if it reduces conflicts or maintains sum-free status with larger size.
        """
        C = list(initial_set)
        C_set = set(C)
        outside = [x for x in S if x not in C_set]
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try each possible swap
            best_swap = None
            best_conflicts = count_conflicts(C_set)
            
            for remove_x in C:
                if time.time() > deadline:
                    break
                new_C_set = C_set - {remove_x}
                
                for add_y in outside:
                    if add_y == remove_x:
                        continue
                    test_set = new_C_set | {add_y}
                    
                    # Check if adding y to (C - {x}) creates violations
                    # Only need to check violations involving y
                    creates_violation = False
                    for c in new_C_set:
                        if y + c in test_set:
                            creates_violation = True
                            break
                        if c - y in test_set:
                            creates_violation = True
                            break
                    if not creates_violation and 2*y in test_set:
                        creates_violation = True
                    
                    if not creates_violation:
                        # This swap maintains sum-free status
                        # Prefer swaps that don't change size (maintain) or we can just take first
                        if best_swap is None:
                            best_swap = (remove_x, add_y)
                        improved = True
                        break
                if improved:
                    break
            
            if best_swap is not None:
                remove_x, add_y = best_swap
                C.remove(remove_x)
                C_set.remove(remove_x)
                C.append(add_y)
                C_set.add(add_y)
                outside = [x for x in outside if x != add_y]
                outside.append(remove_x)
        
        return C
    
    def odd_subset(S):
        return [x for x in S if x % 2 != 0]
    
    def even_upper_half(S):
        """Elements in upper half by value, prefer evens."""
        sorted_S = sorted(S)
        mid = len(sorted_S) // 2
        upper = sorted_S[mid:]
        evens = [x for x in upper if x % 2 == 0]
        odds = [x for x in upper if x % 2 != 0]
        return evens + odds[:max(0, len(upper) - len(evens))]
    
    def large_abs_subset(S):
        """Elements with largest absolute values."""
        sorted_S = sorted(S, key=lambda x: -abs(x))
        return sorted_S[:len(S)//2 + 1]
    
    results = []
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    for S in instances:
        best_set = []
        
        # Baseline: Odds
        odds = odd_subset(S)
        if len(odds) > len(best_set):
            best_set = odds
        
        # Try various greedy removal strategies
        tie_breakers = [
            lambda x: abs(x),      # abs_desc
            lambda x: -abs(x),     # abs_asc
            lambda x: x,           # val_desc
            lambda x: -x,          # val_asc
            lambda x: 0,           # arbitrary
        ]
        
        for weight_func in tie_breakers:
            if time.time() > deadline:
                break
            res = greedy_remove_weighted(S, weight_func, deadline)
            if len(res) > len(best_set):
                best_set = res
        
        # Try greedy add with different orderings
        orderings = [
            lambda x: abs(x),      # increasing abs
            lambda x: -abs(x),     # decreasing abs
            lambda x: x,           # increasing value
            lambda x: -x,          # decreasing value
            lambda x: (abs(x), x), # abs then value
            lambda x: (-abs(x), x),# -abs then value
        ]
        
        for order_key in orderings:
            if time.time() > deadline:
                break
            res = greedy_add(S, order_key, deadline)
            if len(res) > len(best_set):
                best_set = res
        
        # Try upper half / large abs baselines
        for baseline_func in [even_upper_half, large_abs_subset]:
            if time.time() > deadline:
                break
            baseline = baseline_func(S)
            # Clean up baseline to be sum-free
            cleaned = list(baseline)
            cleaned_set = set(cleaned)
            while True:
                if time.time() > deadline:
                    break
                conflicts = get_conflict_elements(cleaned_set)
                if max(conflicts.values()) == 0:
                    break
                max_conf = max(conflicts.values())
                candidates = [x for x, c in conflicts.items() if c == max_conf]
                to_remove = max(candidates, key=lambda x: abs(x))
                cleaned.remove(to_remove)
                cleaned_set.remove(to_remove)
            if len(cleaned) > len(best_set):
                best_set = cleaned
        
        # Local search improvement on best set found so far
        if time.time() <= deadline:
            improved = local_search_swap(S, best_set, deadline)
            if len(improved) > len(best_set):
                best_set = improved
        
        # Try random restarts with perturbed greedy removal
        if time.time() <= deadline:
            rng = random.Random(42)
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Perturb: randomly remove a few elements first, then greedy
                perturbed = list(S)
                for _ in range(min(3, len(perturbed))):
                    if perturbed:
                        idx = rng.randint(0, len(perturbed) - 1)
                        perturbed.pop(idx)
                res = greedy_remove_weighted(perturbed, lambda x: abs(x), deadline)
                if len(res) > len(best_set):
                    best_set = res
        
        results.append(best_set)
        
    return results
