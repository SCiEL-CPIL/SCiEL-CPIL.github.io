import time
import math
import sys
from itertools import combinations

def construct():
    """
    Find a nonempty subset of [1..22] such that all reciprocal subset sums are distinct.
    Maximize cardinality.
    
    Two reciprocal subset sums are equal iff:
        sum_{i in A} 1/i == sum_{j in B} 1/j
    This is equivalent to:
        sum_{i in A \\ B} LCM(1..22)/i == sum_{j in B \\ A} LCM(1..22)/j
    
    So we need a set of integers {w_i = LCM(1..22)/i for i in S} such that no two
    disjoint subsets of {w_i} have the same sum (i.e., the set is dissociated).
    
    Strategy: Use a greedy/backtracking approach with the weight representation.
    We want to maximize |S|. Start with all 22 elements and try to remove as few
    as possible to make the set dissociated.
    
    Actually, let's think differently. We want to find the largest subset S of [1..22]
    such that the weights w_i = LCM(1..22)/i form a dissociated set (no non-trivial
    signed sum equals zero).
    
    We'll use a greedy approach: start with all 22, then iteratively remove elements
    that cause collisions. But this may not be optimal.
    
    Better: Use backtracking to find the maximum cardinality subset.
    
    Let's compute LCM(1..22) and the weights first.
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Compute LCM(1..22)
    lcm_val = 1
    for i in range(1, 23):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
    
    # Weights w_i = LCM(1..22) / i for i in 1..22
    # We need a dissociated subset of these weights
    # i.e., no non-empty subset of signed weights sums to 0
    
    def is_dissociated(weights):
        """Check if a list of positive integers is dissociated (no nontrivial signed sum = 0)"""
        n = len(weights)
        # A set is dissociated iff the only solution to sum c_i * w_i = 0 with c_i in {-1,0,1}
        # is c_i = 0 for all i.
        # Equivalent: all 3^n subset sums (with signs) are distinct, or
        # more efficiently: no two disjoint subsets have the same sum.
        # 
        # For checking: generate all subset sums. If any two distinct subsets have the same sum,
        # the difference gives a non-trivial signed relation.
        # But we need to be careful: two subsets A, B with sum(A) == sum(B) means
        # sum(A\B) - sum(B\A) = 0.
        # So we just need to check if all 2^n subset sums are distinct.
        
        # For n up to ~12-14, 2^n is manageable
        # Use a set of subset sums
        subset_sums = set()
        # Use bit masks for subset sums
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += weights[i]
            if s in subset_sums:
                return False
            subset_sums.add(s)
        return True
    
    # Try to find maximum cardinality subset
    # Start with largest possible and go down
    
    # We have 22 elements indexed 0..21 corresponding to i=1..22
    all_elements = list(range(22))
    weights_all = [lcm_val // (i + 1) for i in range(22)]
    
    best_result = [1]  # Fallback: at least one element always works
    best_cardinality = 1
    
    # Try sizes from 22 down to best_cardinality + 1
    # But 2^22 is too large for full check. Use smarter approach.
    
    # Greedy + local search approach:
    # 1. Start with all 22, find collisions, remove problematic elements
    # 2. Try to add back elements
    
    # Actually, let's use a different approach:
    # The condition is that all subset sums of the weights are distinct.
    # This is equivalent to the weights being a "superincreasing-like" sequence, but not exactly.
    # 
    # Let's try: sort elements by weight and use greedy inclusion.
    
    # Alternative: use the fact that we can check smaller sets.
    # Let's try all subsets of size 12 down to find max dissociated subset.
    
    # But 22 choose 12 is ~6 million, and checking each takes 2^12 = 4096 operations.
    # That's ~25 billion operations, too slow.
    
    # Better: use a smarter search. Start with all 22, detect violations, remove.
    
    def find_violation(weights):
        """Find two distinct subsets with the same sum. Returns (mask1, mask2) or None."""
        n = len(weights)
        subset_sums = {}
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += weights[i]
            if s in subset_sums:
                return (subset_sums[s], mask)
            subset_sums[s] = mask
        return None
    
    # Greedy removal: start with all 22, find violation, remove one of the offending elements
    current = set(range(22))
    
    while time.time() < deadline:
        if not current:
            break
        
        weights_current = [weights_all[i] for i in sorted(current)]
        
        # Check if dissociated
        if len(weights_current) <= 20:  # 2^20 is ~1M, manageable
            violation = find_violation(weights_current)
            if violation is None:
                # Found a valid set
                result = sorted(i + 1 for i in current)
                if len(result) > best_cardinality:
                    best_cardinality = len(result)
                    best_result = result
                    if len(result) == 22:
                        break
                break
            else:
                # Find which elements are in the violating subsets
                mask1, mask2 = violation
                # These are masks into weights_current, which corresponds to sorted(current)
                sorted_current = sorted(current)
                violating_elements = set()
                for i in range(len(weights_current)):
                    if (mask1 | mask2) & (1 << i):
                        violating_elements.add(sorted_current[i])
                
                # Remove the element that appears in the most violations or just remove one
                # Try removing each element in the violation and see which leads to a smaller set
                # For now, just remove the first one
                rem_elem = list(violating_elements)[0]
                current.remove(rem_elem)
        else:
            # For large sets, use a heuristic: check smaller random subsets
            # Or just remove the largest weight (most likely to cause issues)
            # Actually, for large sets, let's just remove elements one by one
            # Remove the element with the largest weight (hardest to fit)
            max_idx = max(current, key=lambda i: weights_all[i])
            current.remove(max_idx)
    
    # If we didn't find a good solution via greedy, try a local search
    if len(best_result) < 15 and time.time() < deadline:
        # Try random restarts with local search
        import random
        random.seed(42)
        
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Start with a random subset of size ~12
            start_size = min(12, 22)
            subset = set(random.sample(range(22), start_size))
            
            # Try to grow
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for candidate in range(22):
                    if candidate in subset:
                        continue
                    new_subset = subset | {candidate}
                    if len(new_subset) > 20:
                        continue
                    weights_new = [weights_all[i] for i in sorted(new_subset)]
                    if find_violation(weights_new) is None:
                        subset = new_subset
                        changed = True
                        if len(subset) > best_cardinality:
                            best_cardinality = len(subset)
                            best_result = sorted(i + 1 for i in subset)
    
    return best_result
