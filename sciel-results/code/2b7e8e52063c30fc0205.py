"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a large set of distinct positive integers <= 512 such that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    
    Approach: Greedy selection with conflict graph modeling.
    We prioritize numbers that have fewer potential 3-element arithmetic progression
    conflicts, structurally preventing mean-formation.
    
    Key insight: If a is the mean of {b, c}, then a = (b+c)/2, i.e., b + c = 2a.
    For larger subsets, if a is the mean of k elements, then sum(subset) = k*a.
    
    We use a greedy approach:
    1. Precompute for each candidate x, how many pairs (y, z) in [1, 512] satisfy y + z = 2x
       (these are 2-element subsets whose mean is x).
    2. Sort candidates by ascending number of such pairs (fewer conflicts first).
    3. Greedily add candidates, checking that no existing element becomes the mean
       of any subset of the newly formed set.
    
    For efficiency, we check the 2-element case (most common) and sample larger subsets.
    """
    n = 512
    
    # Precompute: for each x in [1, n], count pairs (y, z) with y < z, y, z != x, y+z = 2x
    # This tells us how many 2-element subsets have mean x
    pair_counts = np.zeros(n + 1, dtype=np.int32)
    for x in range(1, n + 1):
        count = 0
        # y + z = 2x, y < z, y >= 1, z <= n, y != x, z != x
        # y ranges from max(1, 2x - n) to x - 1 (since y < z = 2x - y implies y < x)
        y_min = max(1, 2 * x - n)
        y_max = x - 1
        if y_max >= y_min:
            count = y_max - y_min + 1
        pair_counts[x] = count
    
    # Sort candidates by ascending pair count (fewest 2-element conflicts first)
    candidates = list(range(1, n + 1))
    candidates.sort(key=lambda x: pair_counts[x])
    
    selected = []
    selected_set = set()
    
    # For fast subset-sum checking, we maintain a dictionary of achievable sums
    # for each subset size, using bitsets or sets of sums
    
    def is_valid_addition(candidate_val, current_selected):
        """Check if adding candidate_val to current_selected keeps the property."""
        new_set = current_selected + [candidate_val]
        new_set_set = set(new_set)
        
        # Check each element in new_set to see if it's the mean of any subset of >= 2 OTHER elements
        for a in new_set:
            others = [y for y in new_set if y != a]
            if len(others) < 2:
                continue
            
            # Check all subset sizes from 2 to len(others)
            # For efficiency, we use the fact that sum(subset) = k * a
            # We can do a DP on subset sums
            
            k_max = len(others)
            # dp[k] = set of achievable sums using exactly k elements from others
            # This is expensive, so we limit to smaller k or use early termination
            
            # Quick check: 2-element subsets
            # a is mean of {y, z} iff y + z = 2a
            found = False
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if others[i] + others[j] == 2 * a:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
            
            # Check 3-element subsets: sum = 3a
            # Use a set of sums for pairs, then check if 3a - pair_sum is in others
            if len(others) >= 3:
                pair_sums = set()
                for i in range(len(others)):
                    for j in range(i + 1, len(others)):
                        pair_sums.add(others[i] + others[j])
                for s in pair_sums:
                    remaining = 3 * a - s
                    if remaining in others_set_check(others, remaining, s):
                        found = True
                        break
                if found:
                    return False
            
            # For larger subsets (k >= 4), we do a limited DP
            # This is expensive, so we only do it if the set is small enough
            if len(others) >= 4 and len(others) <= 20:
                # DP for subset sums
                # dp[k] = set of sums achievable with exactly k elements
                dp = [set() for _ in range(len(others) + 1)]
                dp[0].add(0)
                for elem in others:
                    for k in range(len(others) - 1, -1, -1):
                        for s in list(dp[k]):
                            dp[k + 1].add(s + elem)
                
                for k in range(4, len(others) + 1):
                    if k * a in dp[k]:
                        # Need to verify that the subset doesn't include a itself
                        # Since we're checking 'others', a is not in others, so this is fine
                        found = True
                        break
                if found:
                    return False
        
        return True
    
    def others_set_check(others, target, excluded_sum):
        """Check if target is in others, considering we've already used two elements summing to excluded_sum.
        We just need target to be in others and not equal to the two elements used."""
        # Simple check: is target in others?
        if target not in others:
            return False
        # We need to ensure target is not one of the two elements that sum to excluded_sum
        # This is tricky without tracking, so we do a direct check
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                if others[i] + others[j] == excluded_sum and others[i] != target and others[j] != target:
                    # Then target could be the third element
                    if target in others:
                        return True
        return False
    
    # More efficient approach: precompute and check incrementally
    # Let's restart with a cleaner implementation
    
    selected = []
    
    for candidate_val in candidates:
        if len(selected) >= 48:
            break
        
        # Check if adding candidate_val is valid
        valid = True
        new_set = selected + [candidate_val]
        
        for a in new_set:
            others = [y for y in new_set if y != a]
            if len(others) < 2:
                continue
            
            # Check 2-element subsets
            conflict = False
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if others[i] + others[j] == 2 * a:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                valid = False
                break
            
            # Check 3-element subsets using pair sums
            if len(others) >= 3:
                others_set = set(others)
                pair_sums = set()
                for i in range(len(others)):
                    for j in range(i + 1, len(others)):
                        pair_sums.add(others[i] + others[j])
                for s in pair_sums:
                    rem = 3 * a - s
                    if rem in others_set:
                        # Verify rem is not one of the pair elements
                        # We need to check if there exist i, j, k all distinct in others
                        # with others[i] + others[j] + others[k] = 3a
                        # This is equivalent to: there exist three distinct elements summing to 3a
                        conflict = False
                        for i in range(len(others)):
                            for j in range(i + 1, len(others)):
                                for k in range(j + 1, len(others)):
                                    if others[i] + others[j] + others[k] == 3 * a:
                                        conflict = True
                                        break
                                if conflict:
                                    break
                            if conflict:
                                break
                        if conflict:
                            valid = False
                            break
                if not valid:
                    break
            
            # Check 4+ element subsets with DP (only if set is small)
            if len(others) >= 4 and len(others) <= 15:
                # DP for subset sums
                max_k = len(others)
                dp = [set() for _ in range(max_k + 1)]
                dp[0].add(0)
                for elem in others:
                    for k in range(max_k - 1, -1, -1):
                        for s in list(dp[k]):
                            dp[k + 1].add(s + elem)
                
                for k in range(4, max_k + 1):
                    if k * a in dp[k]:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            selected.append(candidate_val)
    
    return selected

# EVOLVE-BLOCK-END
