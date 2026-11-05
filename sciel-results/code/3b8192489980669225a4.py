"""Fast constructor for mean-free subset problem."""
import time
import random

def _check_zero_sum_subset(diffs, min_size=2):
    """
    Check if there is a subset of 'diffs' of size >= min_size that sums to 0.
    Uses bitset DP for speed.
    """
    n = len(diffs)
    if n < min_size:
        return False
    
    # Calculate bounds for the sums
    min_sum = sum(d for d in diffs if d < 0)
    max_sum = sum(d for d in diffs if d > 0)
    
    # If 0 is not in [min_sum, max_sum], return False (though 0 is always in range if we have +/-)
    # Actually, if all diffs are positive or all negative, 0 is not possible unless 0 is in diffs.
    # But diffs are x - a. If x is not a, 0 is not in diffs.
    
    offset = -min_sum
    size = max_sum - min_sum + 1
    
    # dp[k] is a bitset. Bit i is 1 if sum (i - offset) is achievable with k elements.
    # We only care about sums that can lead to 0 eventually?
    # No, we just need to know if 0 is achievable.
    
    # Optimization: We can prune states that are too far from 0?
    # For now, full range.
    
    dp = [0] * (n + 1)
    dp[0] = 1 << offset  # Sum 0 (which is offset) is achievable with 0 elements
    
    for d in diffs:
        # Shift and OR
        # We need to update dp in reverse order of k to avoid reusing elements
        for k in range(n, 0, -1):
            if dp[k-1]:
                # Shift the bitset by d
                # If d > 0, shift left by d. If d < 0, shift right by -d.
                if d >= 0:
                    dp[k] |= (dp[k-1] << d)
                else:
                    dp[k] |= (dp[k-1] >> (-d))
                # Mask to keep size reasonable? 
                # The bitset can grow. 
                # We only care if bit 'offset' (sum 0) is set.
                # But we need intermediate sums.
                # We can mask to [0, size) to prevent huge ints?
                # The max sum is max_sum, so bit index max_sum + offset.
                # The min sum is min_sum, so bit index min_sum + offset = 0.
                # So the bitset should stay within [0, size).
                # However, Python ints don't have a fixed width, so they just grow.
                # This is efficient enough for size ~ 50k bits.
    
    # Check if sum 0 (index offset) is achievable with k >= min_size
    target_bit = 1 << offset
    for k in range(min_size, n + 1):
        if dp[k] & target_bit:
            return True
            
    return False

def _check_target_sum_subset(diffs, target, min_size=1):
    """
    Check if there is a subset of 'diffs' of size >= min_size that sums to 'target'.
    """
    n = len(diffs)
    if n < min_size:
        return False
    
    min_sum = sum(d for d in diffs if d < 0)
    max_sum = sum(d for d in diffs if d > 0)
    
    if target < min_sum or target > max_sum:
        return False
        
    offset = -min_sum
    size = max_sum - min_sum + 1
    
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for d in diffs:
        for k in range(n, 0, -1):
            if dp[k-1]:
                if d >= 0:
                    dp[k] |= (dp[k-1] << d)
                else:
                    dp[k] |= (dp[k-1] >> (-d))
    
    target_bit = 1 << (target + offset)
    for k in range(min_size, n + 1):
        if dp[k] & target_bit:
            return True
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Start with a base set that is likely valid
    # Powers of 2 are safe
    selected = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    selected_set = set(selected)
    
    # Candidates: all numbers 1..512
    candidates = list(range(1, 513))
    
    # Shuffle to avoid bias, but maybe keep some structure?
    # Let's try sorting by distance from current mean? Or just random?
    # Random is better for exploration.
    random.seed(42)
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        
        # Check if x is valid to add
        valid = True
        
        # 1. Check if x is the mean of any subset of selected
        # Diffs: a - x for a in selected
        diffs_x = [a - x for a in selected]
        if _check_zero_sum_subset(diffs_x, min_size=2):
            valid = False
        
        # 2. Check if any a in selected is the mean of subset of selected + {x}
        if valid:
            for a in selected:
                # Subsets containing x and some from selected \ {a}
                # Sum(S) + x = (|S|+1) a
                # Sum(S) - |S|a = a - x
                # Let S be subset of selected \ {a}
                # Diffs: b - a for b in selected \ {a}
                # Target: a - x
                # Min size of S: 1 (so total size >= 2)
                others = [b for b in selected if b != a]
                if not others:
                    continue
                diffs_a = [b - a for b in others]
                target = a - x
                if _check_target_sum_subset(diffs_a, target, min_size=1):
                    valid = False
                    break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
            
            if len(selected) == 48:
                break
                
    # If we didn't reach 48, try a second pass with different order?
    # Or just return what we have.
    
    return selected[:48]
