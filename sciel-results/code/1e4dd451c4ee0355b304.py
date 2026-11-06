import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # Keep well under 60s limit
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
    
    # smooth_numbers is sorted
    n_smooth = len(smooth_numbers)
    
    # Precompute divisibility matrix for speed? 
    # divs[i][j] = True if smooth_numbers[i] divides smooth_numbers[j]
    # Since n is small (45), a 45x45 matrix is tiny.
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                divs[i][j] = True
            else:
                a = smooth_numbers[i]
                b = smooth_numbers[j]
                if b % a == 0:
                    divs[i][j] = True
                    divs[j][i] = False # b is multiple of a, a does not divide b unless equal
                # Note: if a divides b, b does not divide a (unless a==b)
                # We need "a divides b" or "b divides a" to be False for antichain.
                # So we check if divs[i][j] or divs[j][i] is True.
                # Wait, if a|b, then divs[i][j] is True. 
                # If b|a, then divs[j][i] is True.
                # So if divs[i][j] or divs[j][i] is True, they are comparable.
                # We want them to be incomparable.
    
    # Helper to check if adding 'val_idx' to 'chosen_indices' maintains antichain
    def is_valid_addition(chosen_indices, val_idx):
        for c_idx in chosen_indices:
            if divs[c_idx][val_idx] or divs[val_idx][c_idx]:
                return False
        return True

    answers = [None] * 128
    
    # For each target 1..128
    for target in range(1, 129):
        if time.time() > DEADLINE:
            # Emergency fallback: just return what we have, fill rest with something valid if possible
            # But we expect to finish well before deadline.
            break
            
        found = False
        
        # Try k = 1
        # Check if target is in smooth_numbers
        # Since smooth_numbers is sorted, we can use binary search or just loop (it's small)
        # But we already have the list.
        if target in smooth_numbers:
            # Find index
            # We just need the value
            answers[target - 1] = [target]
            found = True
        else:
            # Try k = 2
            # We need a + b = target, a != b, a, b in smooth_numbers, incomparable
            # Iterate a in smooth_numbers
            # b = target - a
            # if b < a: break (since a is increasing, b is decreasing, we'd have covered it)
            # Actually, just check all a, ensure b is in smooth_numbers and b != a
            # And check antichain condition.
            
            # To be efficient, we can iterate a in smooth_numbers where a < target/2
            for i in range(n_smooth):
                a = smooth_numbers[i]
                if a >= target:
                    break
                b = target - a
                # Check if b is in smooth_numbers
                # We can use a set for O(1) lookup
                # Let's precompute a set
                # But we need the index of b to check divs.
                # Let's create a value-to-index map
                # Do this once outside the loop? Yes.
                
            # Let's restructure to be cleaner.
            # Pre-compute val_to_idx
            val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
            
            # Retry k=2 with map
            for i in range(n_smooth):
                a = smooth_numbers[i]
                if a >= target:
                    break
                b = target - a
                if b < a: 
                    # We've passed the middle, no need to check duplicates
                    # But wait, if a > b, we already checked pair (b, a) when i was index of b?
                    # Yes, because b < a implies index of b < index of a (sorted list).
                    # So we can break if b < a? 
                    # Actually, if a > target/2, then b < target/2 < a.
                    # So we can stop when a > target/2.
                    break
                if b in val_to_idx:
                    j = val_to_idx[b]
                    if i == j:
                        continue
                    # Check antichain
                    if not (divs[i][j] or divs[j][i]):
                        answers[target - 1] = [a, b]
                        found = True
                        break
            
            if found:
                continue
                
            # Try k = 3, 4, ... using backtracking
            # We will write a generic DFS for min cardinality
            
            def find_min_rep(target_val, max_k=10):
                # Returns list of values or None
                # We try k=3, then k=4, etc.
                # Since k=1,2 failed, we start at k=3.
                
                # To speed up, we can use a recursive function that builds the set.
                # State: current_sum, last_idx (to ensure distinct and sorted order to avoid permutations), chosen list
                # Pruning:
                # 1. current_sum > target_val
                # 2. Not enough numbers left to reach target? (Hard to estimate quickly, skip)
                # 3. If we have chosen 'k' numbers and sum == target, check antichain (already done incrementally)
                
                # We want min k.
                # We can just run DFS for k=3. If found, return.
                # If not, run for k=4.
                
                # Optimization: Pre-compute suffix sums? 
                # sum(smooth_numbers[i:]) >= remaining?
                
                for k in range(3, max_k + 1):
                    # Use a flag to stop early if found
                    res = [None]
                    
                    def dfs(start_idx, remaining, chosen):
                        if len(chosen) == k:
                            if remaining == 0:
                                res[0] = list(chosen)
                                return True
                            else:
                                return False
                            
                        # Pruning: if remaining < 0, return False
                        if remaining < 0:
                            return False
                            
                        # Pruning: if we can't fill remaining with the smallest available numbers
                        # min_possible_sum = sum of next (k - len(chosen)) smallest numbers from start_idx onwards
                        # This is a bit expensive to compute dynamically.
                        # Simpler: if remaining is very large, and we have few slots left, maybe impossible?
                        # Let's just rely on the small search space.
                        
                        for i in range(start_idx, n_smooth):
                            val = smooth_numbers[i]
                            if val > remaining:
                                break
                            
                            # Check antichain with chosen
                            valid = True
                            for c_idx in chosen:
                                if divs[c_idx][i] or divs[i][c_idx]:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            chosen.append(i)
                            if dfs(i + 1, remaining - val, chosen):
                                return True
                            chosen.pop()
                            
                    # Start DFS
                    if dfs(0, target_val, []):
                        # Convert indices to values
                        vals = [smooth_numbers[idx] for idx in res[0]]
                        return vals
                    else:
                        # Not found for this k, try next k
                        continue
                        
                return None

            rep = find_min_rep(target)
            if rep:
                answers[target - 1] = rep
            else:
                # Should not happen for 1..128
                # Fallback: just put [target] even if invalid? No, must be valid.
                # But 1..128 are all representable.
                pass

    return answers

# Refinement: The nested function `find_min_rep` is defined inside the loop in the draft above.
# It's better to define it outside or make it efficient.
# Also, the k=2 loop logic with `break` on `b < a` is correct for sorted unique elements.
# Let's refine the code structure for clarity and speed.

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    # 1. Generate smooth numbers
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    n_smooth = len(smooth_numbers)
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # Divisibility matrix
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth_numbers[j] % smooth_numbers[i] == 0:
                    divs[i][j] = True
            else:
                divs[i][j] = True # Self divides self, but we use distinct indices

    answers = [None] * 128
    
    # Helper for backtracking
    def find_min_rep(target_val):
        # k=1
        if target_val in val_to_idx:
            return [target_val]
            
        # k=2
        for i in range(n_smooth):
            a = smooth_numbers[i]
            if a >= target_val:
                break
            b = target_val - a
            if b < a:
                break
            if b in val_to_idx:
                j = val_to_idx[b]
                if i != j and not (divs[i][j] or divs[j][i]):
                    return [a, b]
                    
        # k >= 3
        # We search for k=3, then 4, etc.
        # Max k is likely small.
        for k in range(3, 10):
            found = False
            result = [None]
            
            def dfs(start_idx, remaining, chosen):
                if len(chosen) == k:
                    if remaining == 0:
                        result[0] = list(chosen)
                        return True
                    return False
                    
                if remaining < 0:
                    return False
                    
                # Pruning: if remaining > sum of largest (k-len(chosen)) numbers available from start_idx?
                # Or if remaining < sum of smallest (k-len(chosen)) numbers available from start_idx?
                # Let's add a simple lower bound check.
                # The smallest numbers we can pick are smooth_numbers[start_idx], ..., smooth_numbers[start_idx + k - len(chosen) - 1]
                # If their sum > remaining, we can't reach it? No, we need sum == remaining.
                # If sum of smallest available > remaining, then we can't form remaining with these or larger.
                # So prune.
                
                slots_left = k - len(chosen)
                if slots_left > n_smooth - start_idx:
                    return False
                
                min_sum = 0
                for m in range(slots_left):
                    min_sum += smooth_numbers[start_idx + m]
                if min_sum > remaining:
                    return False
                    
                # Max sum check?
                # If remaining > sum of largest slots_left numbers from end?
                # That's harder to compute dynamically.
                
                for i in range(start_idx, n_smooth):
                    val = smooth_numbers[i]
                    if val > remaining:
                        break
                    
                    # Check antichain
                    valid = True
                    for c_idx in chosen:
                        if divs[c_idx][i] or divs[i][c_idx]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    chosen.append(i)
                    if dfs(i + 1, remaining - val, chosen):
                        return True
                    chosen.pop()
                    
            if dfs(0, target_val, []):
                return [smooth_numbers[idx] for idx in result[0]]
                
        return None

    for target in range(1, 129):
        if time.time() > DEADLINE:
            # Fill remaining with best effort or break
            # But we expect to finish.
            break
        answers[target - 1] = find_min_rep(target)
        
    return answers
