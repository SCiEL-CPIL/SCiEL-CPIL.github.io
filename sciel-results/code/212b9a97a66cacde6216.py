"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
import random
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 22
    # Calculate LCM of 1..22 to convert to integer sums
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Map each x to its integer reciprocal value
    vals = {x: L // x for x in range(1, n + 1)}
    
    best_result = [1]  # fallback
    best_k = 1
    
    def _check_dissociated(v_list):
        """Check if all subset sums of v_list are distinct."""
        if not v_list:
            return True
        sums = {0}
        for v in v_list:
            shifted = {s + v for s in sums}
            if shifted & sums:
                return False
            sums |= shifted
        return True
    
    def _greedy_build(order):
        """Greedy: add elements in given order if they keep the set dissociated."""
        result = []
        result_sums = {0}
        for idx in order:
            v = vals[idx]
            shifted = {s + v for s in result_sums}
            if not (shifted & result_sums):
                result.append(idx)
                result_sums |= shifted
        return result, result_sums
    
    def _dfs_find_k(k, order, deadline):
        """DFS to find a k-element dissociated subset using the given order."""
        n_elem = len(order)
        
        def dfs(start_idx, current, current_sums):
            nonlocal best_result, best_k
            
            if time.time() > deadline:
                return False
            
            if len(current) == k:
                best_result = list(current)
                best_k = k
                return True
            
            remaining_needed = k - len(current)
            if start_idx + remaining_needed > n_elem:
                return False
            
            for i in range(start_idx, n_elem - remaining_needed + 1):
                if time.time() > deadline:
                    return False
                v = vals[order[i]]
                shifted = {s + v for s in current_sums}
                if shifted & current_sums:
                    continue
                new_sums = current_sums | shifted
                new_current = current + [order[i]]
                
                if dfs(i + 1, new_current, new_sums):
                    return True
                    
            return False
        
        return dfs(0, [], {0})
    
    # Try multiple orderings
    orderings = []
    
    # Ordering 1: by value descending (smallest x first)
    orderings.append(sorted(range(1, n + 1), key=lambda x: -vals[x]))
    
    # Ordering 2: by value ascending
    orderings.append(sorted(range(1, n + 1), key=lambda x: vals[x]))
    
    # Ordering 3: by x ascending
    orderings.append(list(range(1, n + 1)))
    
    # Ordering 4: by x descending
    orderings.append(list(range(n, 0, -1)))
    
    # Ordering 5: random
    rng = random.Random(42)
    random_order = list(range(1, n + 1))
    rng.shuffle(random_order)
    orderings.append(random_order)
    
    # Ordering 6: another random
    rng2 = random.Random(123)
    random_order2 = list(range(1, n + 1))
    rng2.shuffle(random_order2)
    orderings.append(random_order2)
    
    # First, try greedy with each ordering to get good starting points
    for order in orderings:
        if time.time() > deadline:
            break
        greedy_set, _ = _greedy_build(order)
        if len(greedy_set) > best_k:
            best_k = len(greedy_set)
            best_result = list(greedy_set)
    
    # Now try DFS from high k down
    for k in range(n, best_k, -1):
        if time.time() > deadline:
            break
        
        found = False
        for order in orderings:
            if time.time() > deadline:
                break
            if _dfs_find_k(k, order, deadline):
                found = True
                break
        
        if found:
            break
    
    # Local search: try to improve by swapping elements
    if time.time() < deadline - 5.0:
        current_set = list(best_result)
        current_k = len(current_set)
        
        # Try to add elements not in the set
        in_set = set(current_set)
        not_in_set = [x for x in range(1, n + 1) if x not in in_set]
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for x in not_in_set:
                if time.time() > deadline - 2.0:
                    break
                # Try adding x
                test_set = current_set + [x]
                if _check_dissociated([vals[y] for y in test_set]):
                    current_set = test_set
                    in_set.add(x)
                    not_in_set.remove(x)
                    improved = True
                    break
        
        # Try swapping: remove one, add one or two
        if time.time() < deadline - 2.0:
            for i, x in enumerate(list(current_set)):
                if time.time() > deadline - 2.0:
                    break
                for y in not_in_set:
                    if time.time() > deadline - 2.0:
                        break
                    test_set = [z for z in current_set if z != x] + [y]
                    if _check_dissociated([vals[z] for z in test_set]):
                        # Same size, but maybe enables further improvements
                        # Only accept if it helps us add more later
                        pass  # Skip swap for now, focus on additions
        
        if len(current_set) > best_k:
            best_k = len(current_set)
            best_result = current_set
    
    return best_result
# EVOLVE-BLOCK-END
