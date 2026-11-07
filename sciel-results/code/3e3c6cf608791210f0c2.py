import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    best_set = []
    
    # We'll use a global variable to track the best found
    # and a recursive search
    
    # To make the search efficient, we rely on the superincreasing property
    # which guarantees unique subset sums.
    # Then we check divisibility.
    
    def is_valid_divisibility(sums):
        """
        sums: a set of integers (subset sums)
        Returns True if no two distinct sums divide each other.
        """
        lst = list(sums)
        n = len(lst)
        # Sort to help with early termination? Not strictly necessary for O(N^2)
        # But checking smaller dividing larger is faster?
        for i in range(n):
            s1 = lst[i]
            if s1 == 0: continue
            for j in range(n):
                if i == j: continue
                s2 = lst[j]
                if s2 == 0: continue
                if s1 > s2:
                    continue # Only check if s1 < s2 to avoid duplicate checks? 
                    # Actually if s1 < s2 and s1 divides s2, it's invalid.
                    # If s1 > s2, we check s2 % s1? No, s2 cannot divide s1 if s2 < s1 unless s1 is multiple of s2.
                    # So we should check if min(s1, s2) divides max(s1, s2).
                small, large = (s1, s2) if s1 < s2 else (s2, s1)
                if large % small == 0:
                    return False
        return True

    def dfs(start_idx, current_set, current_sums):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
        # Try to add more elements
        # The next element must be > sum(current_set) to maintain unique sums
        current_sum = sum(current_set)
        
        # Candidates for next element
        # Must be distinct, in 1..256
        # And > current_sum
        # And > last element (implied by > current_sum if current_sum >= last element? 
        # If current_set is empty, current_sum=0. Next > 0.
        # If current_set=[10], current_sum=10. Next > 10.
        # This ensures sorted order and superincreasing property.
        
        # Optimization: If current_sum >= 256, we can't add any more.
        if current_sum >= 256:
            return
            
        # Iterate possible next elements
        # Start from max(last_element + 1, current_sum + 1)
        # Since we enforce superincreasing, next > current_sum is sufficient for uniqueness.
        # We also need next <= 256.
        
        start_val = current_sum + 1
        if not current_set:
            start_val = 1
        else:
            # Actually, if current_set is [2, 5], sum=7. Next > 7.
            # Is it possible that a smaller element could be added?
            # No, because we build the set in increasing order.
            # If we skip a number, we can't go back.
            # So we just try all values from start_val to 256.
            pass

        for x in range(start_val, 257):
            if time.time() > deadline:
                return
                
            # Check if x is already in current_set? 
            # Since x > sum(current_set) >= any element in current_set, x is distinct.
            
            new_sums = set(current_sums)
            # Add x
            new_sums.add(x)
            # Add x + s for all s in current_sums
            for s in current_sums:
                new_sums.add(x + s)
                
            # Check validity
            if is_valid_divisibility(new_sums):
                dfs(0, current_set + [x], new_sums)
                
    # Initial call
    # We can start with empty set.
    dfs(0, [], {0}) # Wait, subset sums of empty set is {0}. 
    # But the problem says NONEMPTY subset sums.
    # So for empty set, no sums.
    # My `is_valid_divisibility` should handle the set of sums.
    # If I pass {0}, 0 divides nothing? Or is 0 a sum?
    # "Form every NONEMPTY subset sum".
    # So 0 is NOT a sum.
    # So current_sums should NOT contain 0 initially?
    # But when calculating new sums, we use `s in current_sums`.
    # If current_sums is empty, `x + s` loop does nothing.
    # So new_sums = {x}.
    
    # Let's adjust.
    # `current_sums` should be the set of non-empty subset sums.
    
    def dfs_v2(current_set, current_sums):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
        current_sum = sum(current_set)
        if current_sum >= 256:
            return
            
        start_val = current_sum + 1
        if not current_set:
            start_val = 1
            
        for x in range(start_val, 257):
            if time.time() > deadline:
                return
                
            new_sums = set(current_sums)
            new_sums.add(x)
            for s in current_sums:
                new_sums.add(x + s)
                
            if is_valid_divisibility(new_sums):
                dfs_v2(current_set + [x], new_sums)

    dfs_v2([], set())
    
    # Return the best set found
    # Ensure it's a list of ints
    return list(best_set)
