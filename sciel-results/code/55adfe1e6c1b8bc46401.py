"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    
    def is_valid_addition(R_base, new_elements):
        """
        Check if adding new_elements to a set with reachable sums R_base 
        creates a subset sum of 200.
        R_base is an integer bitset.
        """
        target = 200
        # We need to check if target is reachable in R_base U (R_base + subsets of new_elements)
        # This is equivalent to checking if there exists a subset S_new of new_elements
        # such that target - sum(S_new) is in R_base.
        
        n = len(new_elements)
        # Iterate over all non-empty subsets of new_elements? 
        # For small n (1, 2, 3), this is fine.
        # But we can be smarter.
        
        # For n=1: [u]
        # Check target-u in R_base
        
        # For n=2: [u1, u2]
        # Check target-u1, target-u2, target-u1-u2 in R_base
        
        # For n=3: [u1, u2, u3]
        # Check target-ui, target-ui-uj, target-ui-uj-uk in R_base
        
        # Generic approach for small n:
        # Generate all subset sums of new_elements.
        subset_sums = [0]
        for u in new_elements:
            # Add u to all existing subset sums
            current_sums = list(subset_sums)
            for s in current_sums:
                new_s = s + u
                if new_s <= target: # Optimization: if sum > target, it can't help form target (since R_base sums >= 0)
                    subset_sums.append(new_s)
        
        # Check if any (target - s) is in R_base
        for s in subset_sums:
            remainder = target - s
            if remainder < 0:
                continue
            if (R_base >> remainder) & 1:
                return False # Invalid, target is reachable
        
        return True # Valid

    def compute_reachable(S):
        R = 1
        mask = (1 << 201) - 1
        for x in S:
            R |= (R << x)
            R &= mask
        return R

    def greedy(perm):
        selected = []
        R = 1
        mask = (1 << 201) - 1
        for x in perm:
            # Check if adding x makes 200 reachable
            # Equivalent to: is (200-x) in R?
            if 200 - x >= 0:
                if (R >> (200 - x)) & 1:
                    continue
            # Safe to add
            selected.append(x)
            R |= (R << x)
            R &= mask
        return selected

    # 1. Generate initial candidates using different greedy orders
    candidates = []
    
    # Descending
    candidates.append(greedy(range(100, 0, -1)))
    
    # Ascending
    candidates.append(greedy(range(1, 101)))
    
    # Random shuffles
    random.seed(42)
    for _ in range(50):
        perm = list(range(1, 101))
        random.shuffle(perm)
        candidates.append(greedy(perm))
        
    # Pick the best candidate (max cardinality)
    best_S = max(candidates, key=len)
    S = set(best_S)
    U = set(range(1, 101)) - S
    
    # 2. Local Search
    mask = (1 << 201) - 1
    improved = True
    while improved:
        improved = False
        
        # Try 1-for-2 swaps
        for s in list(S):
            S_temp = S - {s}
            R_base = compute_reachable(S_temp)
            
            # Find valid U elements that can be added to S_temp individually
            # Condition: (R_base >> (200-u)) & 1 == 0
            valid_us = []
            for u in U:
                if 200 - u >= 0:
                    if (R_base >> (200 - u)) & 1:
                        continue
                valid_us.append(u)
                
            # Try pairs from valid_us
            for i in range(len(valid_us)):
                u1 = valid_us[i]
                for j in range(i+1, len(valid_us)):
                    u2 = valid_us[j]
                    
                    # Check if u1+u2 combination creates 200
                    # We already know u1 alone and u2 alone are safe.
                    # Check if 200 - u1 - u2 is in R_base
                    rem = 200 - u1 - u2
                    if rem >= 0:
                        if (R_base >> rem) & 1:
                            continue
                    # If rem < 0, it's safe (sum > 200)
                    
                    # Swap
                    S.remove(s)
                    S.add(u1)
                    S.add(u2)
                    U.remove(u1)
                    U.remove(u2)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        
        # Try 2-for-3 swaps
        if not improved:
            # List of pairs in S
            S_list = list(S)
            for i in range(len(S_list)):
                s1 = S_list[i]
                for j in range(i+1, len(S_list)):
                    s2 = S_list[j]
                    
                    S_temp = S - {s1, s2}
                    R_base = compute_reachable(S_temp)
                    
                    valid_us = []
                    for u in U:
                        if 200 - u >= 0:
                            if (R_base >> (200 - u)) & 1:
                                continue
                        valid_us.append(u)
                        
                    # Try triples from valid_us
                    # This is O(N^3) in valid_us. valid_us can be ~60. 60^3 = 216,000.
                    # Might be slow if done for every pair (1000 pairs).
                    # 216,000 * 1000 = 200M ops. Too slow.
                    # We need to prune or limit.
                    
                    # Optimization: Only try if len(valid_us) is small?
                    # Or just try a subset?
                    # Let's limit the search for triples.
                    if len(valid_us) < 30: # Only if small enough
                        for i_u in range(len(valid_us)):
                            u1 = valid_us[i_u]
                            for j_u in range(i_u+1, len(valid_us)):
                                u2 = valid_us[j_u]
                                # Check u1+u2
                                rem12 = 200 - u1 - u2
                                if rem12 >= 0 and (R_base >> rem12) & 1:
                                    continue
                                    
                                for k_u in range(j_u+1, len(valid_us)):
                                    u3 = valid_us[k_u]
                                    
                                    # Check u1+u3
                                    rem13 = 200 - u1 - u3
                                    if rem13 >= 0 and (R_base >> rem13) & 1:
                                        continue
                                        
                                    # Check u2+u3
                                    rem23 = 200 - u2 - u3
                                    if rem23 >= 0 and (R_base >> rem23) & 1:
                                        continue
                                        
                                    # Check u1+u2+u3
                                    rem123 = 200 - u1 - u2 - u3
                                    if rem123 >= 0 and (R_base >> rem123) & 1:
                                        continue
                                        
                                    # All checks passed
                                    S.remove(s1)
                                    S.remove(s2)
                                    S.add(u1)
                                    S.add(u2)
                                    S.add(u3)
                                    U.remove(u1)
                                    U.remove(u2)
                                    U.remove(u3)
                                    improved = True
                                    break
                                if improved: break
                            if improved: break
                        if improved: break
                    if improved: break
                    
    return list(S)
# EVOLVE-BLOCK-END
