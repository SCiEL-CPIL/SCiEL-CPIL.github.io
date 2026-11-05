import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Known good solution: {66, ..., 100}, size 35
    # Max sum of 2 elements is 99+100=199.
    # Min sum of 3 elements is 66+67+68=201.
    # Thus, no subset sums to 200.
    best_set = set(range(66, 101))
    best_size = 35

    def is_valid(s):
        """Check if no subset of s sums to 200."""
        sums = {0}
        for x in s:
            new = {y + x for y in sums if y + x <= 200}
            if 200 in new:
                return False
            sums |= new
        return True

    def compute_reachable(s):
        """Compute all subset sums up to 200 for set s."""
        reachable = {0}
        for x in s:
            new_sums = {y + x for y in reachable if y + x <= 200}
            reachable |= new_sums
        return reachable

    # Precompute all hyperedges (subsets of [1..100] summing to 200)
    # This is expensive, so we limit the size of subsets considered or use a smarter generation.
    # For N=100 and target=200, subsets of size > 4 are rare for large elements but possible for small.
    # However, generating ALL subsets is 2^100. We cannot do that.
    # Instead, we generate "conflicts" on the fly or use the greedy heuristic described.
    
    # Strategy: Greedy heuristic based on conflict density.
    # We want to select elements to maximize cardinality while avoiding sum=200.
    # This is equivalent to finding a large set with no subset summing to 200.
    
    # Let's try a different approach: Start with all elements, and remove elements that cause conflicts.
    # Or, start with empty, and add elements that don't create a conflict.
    
    # The "conflict density" idea:
    # An element x is "dangerous" if adding it to the current set S creates a subset summing to 200.
    # This happens if there is a subset T of S such that sum(T) + x = 200, i.e., sum(T) = 200 - x.
    
    # We will use the greedy construction with multiple orders, as in the current program,
    # but enhance it with a local search that uses the "swap" logic more effectively.
    
    # Additional Idea: Try to find a set of size 36 or more.
    # If we take {1..100} and remove some elements, can we keep more than 35?
    # The set {66..100} is a "large" interval.
    # What about mixing small and large?
    # If we include small numbers, we have to be careful with the large numbers.
    
    # Let's try a randomized local search starting from the known good solution.
    # Move types:
    # 1. Add one element.
    # 2. Remove one, add two.
    # 3. Remove two, add three.
    # 4. Remove k, add k+1.
    
    # To speed up the "can we add x" check, we maintain the set of reachable sums.
    
    def try_add(s, reachable, x):
        """Try to add x to s. Returns (new_s, new_reachable, success)."""
        if x in s:
            return s, reachable, False
        if (200 - x) in reachable:
            return s, reachable, False
        new_sums = {y + x for y in reachable if y + x <= 200}
        if 200 in new_sums:
            return s, reachable, False
        new_reachable = reachable | new_sums
        new_s = s | {x}
        return new_s, new_reachable, True

    def remove_and_recompute(s, reachable, x):
        """Remove x from s and recompute reachable sums."""
        s = s - {x}
        reachable = compute_reachable(s)
        return s, reachable

    def local_search(initial_set, max_iters=2000):
        s = set(initial_set)
        reachable = compute_reachable(s)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to add a single element
            # Sort candidates by value? Smaller values might be safer? Or larger?
            # Let's try random order for diversity.
            candidates = [x for x in range(1, 101) if x not in s]
            random.shuffle(candidates)
            for x in candidates:
                new_s, new_reachable, ok = try_add(s, reachable, x)
                if ok:
                    s = new_s
                    reachable = new_reachable
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try to swap 1 out, 2 in
            # This is O(N^3) roughly if done naively.
            # We can try to remove one element, then try to add two.
            elements_in_s = list(s)
            random.shuffle(elements_in_s)
            for x_out in elements_in_s:
                if time.time() > deadline:
                    break
                
                s_temp, reachable_temp = remove_and_recompute(s, reachable, x_out)
                
                # Now try to add two elements
                candidates = [x for x in range(1, 101) if x not in s_temp and x != x_out]
                random.shuffle(candidates)
                
                added = []
                s_curr = s_temp
                r_curr = reachable_temp
                
                for x1 in candidates:
                    new_s1, new_r1, ok1 = try_add(s_curr, r_curr, x1)
                    if ok1:
                        s_curr = new_s1
                        r_curr = new_r1
                        added.append(x1)
                        break
                
                if len(added) == 1:
                    for x2 in candidates:
                        if x2 in s_curr:
                            continue
                        new_s2, new_r2, ok2 = try_add(s_curr, r_curr, x2)
                        if ok2:
                            s_curr = new_s2
                            r_curr = new_r2
                            added.append(x2)
                            break
                
                if len(added) == 2:
                    s = s_curr
                    reachable = r_curr
                    improved = True
                    break
            
            if improved:
                continue
            
            # 3. Try to swap 2 out, 3 in
            # This is more expensive. Only try if we have time.
            if time.time() > deadline:
                break
                
            elements_in_s = list(s)
            random.shuffle(elements_in_s)
            
            # Limit the number of pairs to try to save time
            pairs_tried = 0
            for i in range(len(elements_in_s)):
                for j in range(i+1, len(elements_in_s)):
                    if time.time() > deadline:
                        break
                    if pairs_tried > 50: # Limit pairs
                        break
                    pairs_tried += 1
                    
                    x1_out = elements_in_s[i]
                    x2_out = elements_in_s[j]
                    
                    s_temp, reachable_temp = remove_and_recompute(s, reachable, x1_out)
                    s_temp, reachable_temp = remove_and_recompute(s_temp, reachable_temp, x2_out)
                    
                    candidates = [x for x in range(1, 101) if x not in s_temp and x != x1_out and x != x2_out]
                    random.shuffle(candidates)
                    
                    added = []
                    s_curr = s_temp
                    r_curr = reachable_temp
                    
                    for x1 in candidates:
                        new_s1, new_r1, ok1 = try_add(s_curr, r_curr, x1)
                        if ok1:
                            s_curr = new_s1
                            r_curr = new_r1
                            added.append(x1)
                            break
                    
                    if len(added) == 1:
                        for x2 in candidates:
                            if x2 in s_curr:
                                continue
                            new_s2, new_r2, ok2 = try_add(s_curr, r_curr, x2)
                            if ok2:
                                s_curr = new_s2
                                r_curr = new_r2
                                added.append(x2)
                                break
                    
                    if len(added) == 2:
                        for x3 in candidates:
                            if x3 in s_curr:
                                continue
                            new_s3, new_r3, ok3 = try_add(s_curr, r_curr, x3)
                            if ok3:
                                s_curr = new_s3
                                r_curr = new_r3
                                added.append(x3)
                                break
                    
                    if len(added) == 3:
                        s = s_curr
                        reachable = r_curr
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # No improvement found in this iteration, break to avoid infinite loop
                # or try random perturbation?
                # For now, break.
                break
                
        return s

    # Run local search on the known good solution
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iters=1000)
        if len(improved_set) > best_size and is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set

    # Try greedy construction with different orders
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 50, -1)) + list(range(50, 0, -1))
    ]
    
    # Add random orders
    for _ in range(10):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        s = set()
        reachable = {0}
        
        for x in order:
            new_s, new_reachable, ok = try_add(s, reachable, x)
            if ok:
                s = new_s
                reachable = new_reachable
        
        if len(s) > best_size:
            if is_valid(s):
                best_size = len(s)
                best_set = s
        
        # If this greedy result is close to best, try local search on it
        if len(s) >= best_size - 2 and time.time() < deadline - 5:
            improved_set = local_search(s, max_iters=200)
            if len(improved_set) > best_size and is_valid(improved_set):
                best_size = len(improved_set)
                best_set = improved_set

    # Final validation
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(66, 101))
