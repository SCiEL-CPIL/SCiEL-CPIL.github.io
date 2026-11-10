"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_valid_subset(subset):
    """
    Check if subset is reciprocal-subsum-free.
    For each x in subset, 1/x must NOT equal sum of 1/y for any non-empty subset of subset\{x}.
    """
    n = len(subset)
    # Precompute LCM of all elements for efficient integer arithmetic
    # Actually, we need LCM per check or a global approach.
    # Let's use a global LCM for the entire set [1..24] to simplify, or compute per check.
    # Since n is small (max 24), and we're checking subsets, let's optimize.
    
    # For a given subset, for each x in subset:
    #   Let S = subset - {x}
    #   Check if there exists T subset of S such that sum(1/y for y in T) == 1/x
    #   Multiply by L = LCM(S union {x}):
    #   L/x == sum(L/y for y in T)
    
    # We can precompute the LCM of the entire universe [1..24] to avoid repeated LCM calcs?
    # LCM(1..24) is large but manageable.
    # LCM(1..24) = 5354228880
    # Using a fixed LCM for all checks in a single subset evaluation is fine.
    
    LCM_UNIVERSE = 5354228880
    
    for x in subset:
        others = [y for y in subset if y != x]
        if not others:
            continue
        
        target_int = LCM_UNIVERSE // x
        # We need to check if target_int is a subset sum of {LCM_UNIVERSE // y for y in others}
        
        # Meet-in-the-middle for subset sum
        # n_others <= 23
        # If n_others is small, brute force is fine.
        # If n_others is large, meet-in-the-middle.
        
        values = [LCM_UNIVERSE // y for y in others]
        n_vals = len(values)
        
        if n_vals <= 20:
            # Brute force with bitmasks or simple recursion
            # Actually, 2^20 is 1 million, which is fast enough in Python if optimized, but might be slow in a loop.
            # Let's use a set-based dynamic programming for subset sums if values are small?
            # Values are up to 5354228880, so DP array is not possible.
            # Meet-in-the-middle is better for n > 15.
            
            if n_vals <= 15:
                # Brute force
                possible_sums = {0}
                for v in values:
                    new_sums = set()
                    for s in possible_sums:
                        new_sums.add(s + v)
                    possible_sums |= new_sums
                if target_int in possible_sums:
                    return False
            else:
                # Meet-in-the-middle
                half = n_vals // 2
                left_vals = values[:half]
                right_vals = values[half:]
                
                left_sums = {0}
                for v in left_vals:
                    new_sums = set()
                    for s in left_sums:
                        new_sums.add(s + v)
                    left_sums |= new_sums
                
                right_sums = {0}
                for v in right_vals:
                    new_sums = set()
                    for s in right_sums:
                        new_sums.add(s + v)
                    right_sums |= new_sums
                
                # Check if any left_sum + right_sum == target_int
                # Convert one to set for O(1) lookup
                right_set = set(right_sums)
                for ls in left_sums:
                    if (target_int - ls) in right_set:
                        return False
        else:
            # Meet-in-the-middle
            half = n_vals // 2
            left_vals = values[:half]
            right_vals = values[half:]
            
            left_sums = {0}
            for v in left_vals:
                new_sums = set()
                for s in left_sums:
                    new_sums.add(s + v)
                left_sums |= new_sums
            
            right_sums = {0}
            for v in right_vals:
                new_sums = set()
                for s in right_sums:
                    new_sums.add(s + v)
                right_sums |= new_sums
            
            right_set = set(right_sums)
            for ls in left_sums:
                if (target_int - ls) in right_set:
                    return False
                    
    return True

def _branch_and_bound(start, current_subset, best_subset, deadline):
    """
    Recursive branch and bound to find max cardinality valid subset.
    """
    if time.time() > deadline:
        return best_subset
    
    if len(current_subset) > len(best_subset):
        best_subset = list(current_subset)
    
    # Pruning: if remaining elements can't improve size, return
    # (Not implemented here, but can be added)
    
    for i in range(start, 25):
        if time.time() > deadline:
            break
        candidate = current_subset + [i]
        # Check validity incrementally? 
        # Full check is expensive. 
        # We can check if adding i violates the condition for i itself or for existing elements.
        # Condition for i: 1/i != sum(1/y for y in T subset of current_subset)
        # Condition for existing x: 1/x != sum(1/y for y in T subset of (current_subset + [i]) - {x})
        # The latter is hard to check incrementally without re-evaluating all subsets.
        
        # Given the small universe, let's just check full validity for the new candidate.
        # But this is O(2^n) per check, which is slow.
        
        # Optimization: Only check if adding i creates a violation involving i, or if it allows a new subset sum for existing elements.
        # Actually, if the current_subset is valid, adding i can only create violations:
        # 1. 1/i = sum(1/y for y in T subset of current_subset)
        # 2. For some x in current_subset, 1/x = sum(1/y for y in T subset of (current_subset - {x}) union {i})
        #    This means 1/x = sum(1/y for y in T1) + 1/i, where T1 subset of current_subset - {x}
        #    => 1/x - 1/i = sum(1/y for y in T1)
        
        # Let's implement these incremental checks.
        
        # Check 1: Does 1/i equal sum of reciprocals of a subset of current_subset?
        if current_subset:
            target_int_i = LCM_UNIVERSE // i
            values_curr = [LCM_UNIVERSE // y for y in current_subset]
            # Subset sum check for target_int_i in values_curr
            # Use same meet-in-the-middle or brute force
            n_curr = len(values_curr)
            found = False
            if n_curr <= 15:
                possible_sums = {0}
                for v in values_curr:
                    new_sums = set()
                    for s in possible_sums:
                        new_sums.add(s + v)
                    possible_sums |= new_sums
                if target_int_i in possible_sums:
                    found = True
            else:
                half = n_curr // 2
                left_vals = values_curr[:half]
                right_vals = values_curr[half:]
                
                left_sums = {0}
                for v in left_vals:
                    new_sums = set()
                    for s in left_sums:
                        new_sums.add(s + v)
                    left_sums |= new_sums
                
                right_sums = {0}
                for v in right_vals:
                    new_sums = set()
                    for s in right_sums:
                        new_sums.add(s + v)
                    right_sums |= new_sums
                
                right_set = set(right_sums)
                for ls in left_sums:
                    if (target_int_i - ls) in right_set:
                        found = True
                        break
            
            if found:
                continue # i cannot be added
        
        # Check 2: For each x in current_subset, does adding i allow 1/x to be formed?
        # 1/x = sum(1/y for y in T) where T subset of (current_subset - {x}) union {i}
        # This implies either:
        # a) i not in T: already valid since current_subset is valid.
        # b) i in T: 1/x = 1/i + sum(1/y for y in T') where T' subset of current_subset - {x}
        #    => 1/x - 1/i = sum(1/y for y in T')
        #    => (i - x) / (x*i) = sum(1/y for y in T')
        
        for x in current_subset:
            if time.time() > deadline:
                return best_subset
            
            # Calculate target for T'
            # 1/x - 1/i = (i - x) / (x * i)
            # We need to check if (i - x) / (x * i) can be formed by sum of 1/y for y in T' subset of current_subset - {x}
            
            # Convert to integer with LCM_UNIVERSE
            # LCM * (i - x) / (x * i) must be an integer?
            # LCM is divisible by x and i, so LCM/(x*i) is integer.
            # Let K = LCM_UNIVERSE // (x * i)
            # Target_int = K * (i - x)
            # Values = [LCM_UNIVERSE // y for y in current_subset if y != x]
            
            if i == x:
                continue # x is in current_subset, i is new, so i != x always in this loop context? 
                         # Wait, i is being added, x is in current_subset. So i != x.
            
            K = LCM_UNIVERSE // (x * i)
            target_int = K * (i - x)
            
            if target_int < 0:
                continue # Sum of reciprocals is positive, so if target < 0, impossible.
            
            values_prime = [LCM_UNIVERSE // y for y in current_subset if y != x]
            
            # Check if target_int is a subset sum of values_prime
            n_prime = len(values_prime)
            found_prime = False
            if n_prime <= 15:
                possible_sums = {0}
                for v in values_prime:
                    new_sums = set()
                    for s in possible_sums:
                        new_sums.add(s + v)
                    possible_sums |= new_sums
                if target_int in possible_sums:
                    found_prime = True
            else:
                half = n_prime // 2
                left_vals = values_prime[:half]
                right_vals = values_prime[half:]
                
                left_sums = {0}
                for v in left_vals:
                    new_sums = set()
                    for s in left_sums:
                        new_sums.add(s + v)
                    left_sums |= new_sums
                
                right_sums = {0}
                for v in right_vals:
                    new_sums = set()
                    for s in right_sums:
                        new_sums.add(s + v)
                    right_sums |= new_sums
                
                right_set = set(right_sums)
                for ls in left_sums:
                    if (target_int - ls) in right_set:
                        found_prime = True
                        break
            
            if found_prime:
                break # Violation found, i cannot be added with current_subset
        
        if time.time() > deadline:
            break
        
        # If we didn't break, the addition is valid
        _branch_and_bound(i + 1, candidate, best_subset, deadline)
        
        # Backtrack
        # (No explicit backtrack needed as we pass copies)
        
    return best_subset

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Try to find a large valid subset
    # Start with empty set
    best_subset = []
    
    # We can try to build up from large sets or use B&B
    # B&B from empty set might be slow if it explores many invalid branches early.
    # Let's try a heuristic: start with a known good set and try to add elements?
    # Or just run B&B with pruning.
    
    # The universe is 1..24.
    # Let's try to find the maximum size.
    # Known result for small n?
    # For n=24, the maximum size is likely around 10-15?
    
    # Let's use the B&B function.
    # To speed up, we can order elements. Larger elements have smaller reciprocals,
    # so they are less likely to form sums.
    # Let's iterate i from 24 down to 1?
    # The B&B function iterates from start to 25.
    # We can modify it to iterate in a specific order.
    
    # Let's create a modified B&B that tries larger elements first.
    
    def _branch_and_bound_ordered(start, current_subset, best_subset, deadline, order):
        if time.time() > deadline:
            return best_subset
        
        if len(current_subset) > len(best_subset):
            best_subset = list(current_subset)
        
        # Find next element to try
        # order is a list of elements to consider
        # We need to map start index to actual element
        # This is getting complicated. Let's just stick to 1..24 order for simplicity,
        # or reverse it.
        
        # Let's try reverse order: 24, 23, ..., 1
        # But the function signature uses start index.
        
        # Let's just call the original B&B but with a twist:
        # We'll manually drive the search.
        
        # Actually, let's just use the simple B&B but with a time limit.
        # If it's too slow, we'll fall back to a heuristic.
        
        # For now, let's just try to add elements greedily?
        # No, B&B is better for optimality.
        
        # Let's try a different approach:
        # Generate all subsets of size k, starting from k=24 down to 1,
        # and check if any is valid.
        # This is 2^24 which is 16 million. Too slow in Python.
        
        # B&B is the way.
        
        # Let's refine the B&B to use the incremental checks we defined.
        
        pass

    # Let's implement a cleaner B&B with the incremental checks.
    
    LCM_UNIVERSE = 5354228880
    
    def check_addition(i, current_subset):
        """Check if i can be added to current_subset."""
        if not current_subset:
            return True
        
        # Check 1: 1/i = sum(1/y for y in T subset of current_subset)
        target_int_i = LCM_UNIVERSE // i
        values_curr = [LCM_UNIVERSE // y for y in current_subset]
        n_curr = len(values_curr)
        
        found = False
        if n_curr <= 15:
            possible_sums = {0}
            for v in values_curr:
                new_sums = set()
                for s in possible_sums:
                    new_sums.add(s + v)
                possible_sums |= new_sums
            if target_int_i in possible_sums:
                found = True
        else:
            half = n_curr // 2
            left_vals = values_curr[:half]
            right_vals = values_curr[half:]
            
            left_sums = {0}
            for v in left_vals:
                new_sums = set()
                for s in left_sums:
                    new_sums.add(s + v)
                left_sums |= new_sums
            
            right_sums = {0}
            for v in right_vals:
                new_sums = set()
                for s in right_sums:
                    new_sums.add(s + v)
                right_sums |= new_sums
            
            right_set = set(right_sums)
            for ls in left_sums:
                if (target_int_i - ls) in right_set:
                    found = True
                    break
        
        if found:
            return False
        
        # Check 2: For each x in current_subset, does adding i allow 1/x to be formed?
        for x in current_subset:
            if i == x:
                continue
            
            # 1/x - 1/i = sum(1/y for y in T' subset of current_subset - {x})
            # (i - x) / (x * i) = sum(1/y for y in T')
            
            K = LCM_UNIVERSE // (x * i)
            target_int = K * (i - x)
            
            if target_int < 0:
                continue
            
            values_prime = [LCM_UNIVERSE // y for y in current_subset if y != x]
            n_prime = len(values_prime)
            
            found_prime = False
            if n_prime <= 15:
                possible_sums = {0}
                for v in values_prime:
                    new_sums = set()
                    for s in possible_sums:
                        new_sums.add(s + v)
                    possible_sums |= new_sums
                if target_int in possible_sums:
                    found_prime = True
            else:
                half = n_prime // 2
                left_vals = values_prime[:half]
                right_vals = values_prime[half:]
                
                left_sums = {0}
                for v in left_vals:
                    new_sums = set()
                    for s in left_sums:
                        new_sums.add(s + v)
                    left_sums |= new_sums
                
                right_sums = {0}
                for v in right_vals:
                    new_sums = set()
                    for s in right_sums:
                        new_sums.add(s + v)
                    right_sums |= new_sums
                
                right_set = set(right_sums)
                for ls in left_sums:
                    if (target_int - ls) in right_set:
                        found_prime = True
                        break
            
            if found_prime:
                return False
        
        return True

    def bnb(start, current_subset, best_subset, deadline):
        if time.time() > deadline:
            return best_subset
        
        if len(current_subset) > len(best_subset):
            best_subset = list(current_subset)
        
        # Try adding elements from start to 24
        # To find larger sets faster, maybe try larger elements first?
        # Let's try 24 down to start.
        for i in range(24, start - 1, -1):
            if time.time() > deadline:
                break
            
            if check_addition(i, current_subset):
                new_subset = current_subset + [i]
                bnb(i - 1, new_subset, best_subset, deadline)
        
        return best_subset

    best_subset = bnb(0, [], [], deadline)
    
    if not best_subset:
        # Fallback
        best_subset = [1]
        
    return best_subset

# EVOLVE-BLOCK-END
