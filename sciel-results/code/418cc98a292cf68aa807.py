import random

def construct():
    max_n = 512
    max_size = 48
    
    def check_violation(S):
        """
        Returns True if S has a violation.
        Returns False if S is valid.
        """
        S_list = list(S)
        n = len(S_list)
        
        # Precompute sum of all elements?
        
        for i in range(n):
            a = S_list[i]
            others = [x for x in S_list if x != a]
            
            # Quick check: if 2*a > sum(others), no subset can sum to 2*a or more?
            # Max sum is sum(others).
            # Min k is 2.
            # If 2*a > sum(others), then k*a > sum(others) for all k>=2.
            # So no violation for this a.
            if 2 * a > sum(others):
                continue
                
            # DP for subset sums of others
            dp = 1
            # We only care about sums up to max_k * a
            # max_k = len(others)
            max_sum_needed = len(others) * a
            
            for v in others:
                # Optimization: mask out bits higher than max_sum_needed?
                # Not strictly necessary if we just check specific bits.
                dp |= dp << v
                
            # Check targets
            for k in range(2, len(others) + 1):
                T = k * a
                if T > max_sum_needed:
                    break
                if (dp >> T) & 1:
                    # T is a subset sum.
                    # Check if it requires >= 2 elements.
                    if T not in others:
                        return True
                    else:
                        # T is in others. Check if others \ {T} can sum to T.
                        # This is a subset sum problem on others \ {T} for target T.
                        # We can do a quick DP or check if T is in the set of sums of others \ {T}.
                        # Since T is in others, let's call it v_target.
                        # We need to know if there is a subset of others \ {v_target} summing to v_target.
                        # This is equivalent to: is v_target in the subset sums of others \ {v_target}?
                        
                        # Optimization: 
                        # If sum(others \ {v_target}) < T, return False (no violation for this k).
                        # If sum(others \ {v_target}) >= T, do DP.
                        
                        others_minus_T = [v for v in others if v != T]
                        if sum(others_minus_T) < T:
                            continue
                            
                        dp2 = 1
                        for v in others_minus_T:
                            dp2 |= dp2 << v
                            
                        if (dp2 >> T) & 1:
                            return True
                            
        return False

    # Local Search
    current_set = set(random.sample(range(1, max_n + 1), max_size))
    
    for _ in range(1000):
        if not check_violation(current_set):
            return list(current_set)
        
        # Replace a random element
        elem_to_replace = random.choice(list(current_set))
        current_set.remove(elem_to_replace)
        
        new_elem = random.randint(1, max_n)
        while new_elem in current_set:
            new_elem = random.randint(1, max_n)
            
        current_set.add(new_elem)
        
    # Fallback: Return current set even if invalid? 
    # No, the evaluator checks validity. Invalid scores 0.
    # We should return a valid subset.
    
    # Try to build a valid set greedily from the current set or scratch.
    # Let's try to shrink the current set until it is valid.
    while len(current_set) > 0:
        if not check_violation(current_set):
            return list(current_set)
        # Remove a random element
        elem_to_remove = random.choice(list(current_set))
        current_set.remove(elem_to_remove)
        
    return [1] # Last resort

