import time
import itertools
import math

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Precompute LCM of 1..24
    L = 1
    for i in range(1, 25):
        L = _lcm(L, i)
        
    # Precompute bad masks for all x using integer arithmetic
    bad_masks = {}
    for x in range(1, 25):
        target = L // x
        # Elements available: y in 1..24, y != x
        # Values: L // y
        # We want subsets of these values summing to target.
        
        elems = []
        for y in range(1, 25):
            if y != x:
                elems.append((L // y, y))
        
        # Sort by value descending
        elems.sort(key=lambda p: p[0], reverse=True)
        
        n = len(elems)
        # Suffix sums for pruning
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + elems[i][0]
            
        bad_masks_x = []
        
        def dfs(start_idx, current_sum, current_mask):
            if current_sum == target:
                if current_mask != 0:
                    bad_masks_x.append(current_mask)
                return
            
            if current_sum > target:
                return
            
            if start_idx >= n:
                return
            
            if current_sum + suffix_sums[start_idx] < target:
                return
            
            for i in range(start_idx, n):
                val, y = elems[i]
                new_sum = current_sum + val
                if new_sum > target:
                    continue
                dfs(i + 1, new_sum, current_mask | (1 << (y - 1)))
        
        dfs(0, 0, 0)
        bad_masks[x] = bad_masks_x
        
    # Optimization: If time is running out, just return a known valid small set
    # e.g., {24} is always valid? 
    # 1/24. Can it be sum of others? Min sum of 2 others is 1/23+1/22 > 1/24.
    # So {24} is valid.
    # {23, 24}? 1/23 sum of others? Only 24. 1/24 != 1/23.
    # 1/24 sum of others? Only 23. 1/23 != 1/24.
    # So {23, 24} is valid.
    # We can build a greedy valid set quickly as a fallback.
    
    def is_valid_mask(mask):
        # Check for each x in mask
        for x in range(1, 25):
            if (mask >> (x - 1)) & 1:
                if not bad_masks[x]:
                    continue
                # Check if any bad mask for x is subset of mask
                for bm in bad_masks[x]:
                    if (bm & mask) == bm:
                        return False
        return True

    best_mask = 0
    best_size = 0
    
    # Try sizes from 24 down to 1
    # We can stop if we find a valid set of a certain size?
    # The task says "Maximize cardinality". So we should find the max.
    # But we must return *a* set. If we find size 24, that's max possible.
    # If we find size 23, we should check if 24 is possible? 
    # Actually, just iterate down. The first valid set found at size k is a candidate.
    # But we need the MAX size. 
    # So we should check 24, then 23, etc.
    # If 24 is invalid, check 23. If 23 is valid, is it possible that a size 24 set exists? No, we already checked 24.
    # So the first valid size we find is the maximum.
    
    for k in range(24, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Generate combinations of size k
        # For k=24, 1 comb. k=23, 24 combs. k=22, 276. k=21, 2024.
        # This is very fast.
        
        found = False
        for comb in itertools.combinations(range(1, 25), k):
            mask = 0
            for x in comb:
                mask |= (1 << (x - 1))
                
            if is_valid_mask(mask):
                # Found a valid set of size k. Since we iterate k downwards, this is the max.
                return list(comb)
                
        # If no valid set of size k found, continue to k-1
        
    # Fallback: Greedy construction if combinatorial search fails or times out
    current_mask = 0
    # Try to add elements in some order
    # Order: Larger elements first? Smaller?
    # Let's try descending order (24, 23, ...)
    for x in range(24, 0, -1):
        trial = current_mask | (1 << (x - 1))
        if is_valid_mask(trial):
            current_mask = trial
            
    result = [x for x in range(1, 25) if (current_mask >> (x - 1)) & 1]
    
    # Ensure non-empty
    if not result:
        result = [24]
        
    return result
