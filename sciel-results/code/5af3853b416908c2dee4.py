import time
from fractions import Fraction
from functools import reduce

def _lcm(a, b):
    from math import gcd
    return a * b // gcd(a, b)

def _precompute_bad_masks():
    """
    For each x in 1..24, find all non-empty subsets of {1..24}\{x} 
    whose reciprocals sum to 1/x.
    Returns dict: x -> list of bitmasks (bit i set means element i+1 is in subset)
    """
    bad_masks = {x: [] for x in range(1, 25)}
    
    for x in range(1, 25):
        target = Fraction(1, x)
        # Available elements: all except x
        elems = [y for y in range(1, 25) if y != x]
        # Sort by 1/y descending for better pruning
        elems.sort(key=lambda y: -y)  # smaller y means larger 1/y
        
        n = len(elems)
        # Precompute suffix sums of 1/y for pruning
        suffix_sums = [Fraction(0)] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, elems[i])
        
        found_masks = []
        
        def dfs(idx, current_sum, current_mask):
            if current_sum == target:
                if current_mask != 0:
                    found_masks.append(current_mask)
                return
            
            if current_sum > target:
                return
            
            if idx >= n:
                return
            
            # Prune: if we can't reach target even with all remaining
            if current_sum + suffix_sums[idx] < target:
                return
            
            # Try including elems[idx]
            val = Fraction(1, elems[idx])
            new_sum = current_sum + val
            if new_sum <= target:
                dfs(idx + 1, new_sum, current_mask | (1 << (elems[idx] - 1)))
            
            # Try excluding elems[idx]
            dfs(idx + 1, current_sum, current_mask)
        
        dfs(0, Fraction(0), 0)
        bad_masks[x] = found_masks
    
    return bad_masks

def _is_valid_mask(mask, bad_masks):
    """Check if mask represents a valid set."""
    for x in range(1, 25):
        if (mask >> (x - 1)) & 1:
            for bm in bad_masks[x]:
                if (bm & mask) == bm:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin for 90s limit
    
    # Precompute bad masks
    bad_masks = _precompute_bad_masks()
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy from largest to smallest
    current = 0
    for c in range(24, 0, -1):
        if time.time() > deadline - 5:
            break
        bit = 1 << (c - 1)
        trial = current | bit
        if _is_valid_mask(trial, bad_masks):
            current = trial
    if bin(current).count('1') > best_size:
        best_size = bin(current).count('1')
        best_mask = current
    
    # Strategy 2: Greedy from smallest to largest
    current = 0
    for c in range(1, 25):
        if time.time() > deadline - 5:
            break
        bit = 1 << (c - 1)
        trial = current | bit
        if _is_valid_mask(trial, bad_masks):
            current = trial
    if bin(current).count('1') > best_size:
        best_size = bin(current).count('1')
        best_mask = current
    
    # Strategy 3: Start with large seed and extend
    seeds = [
        list(range(13, 25)),
        list(range(12, 25)),
        list(range(11, 25)),
        list(range(10, 25)),
        list(range(14, 25)),
        list(range(9, 25)),
    ]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        current = 0
        for c in seed:
            current |= (1 << (c - 1))
        
        # Check if seed is valid
        if not _is_valid_mask(current, bad_masks):
            continue
        
        # Try to add elements not in seed
        for c in range(1, 25):
            if time.time() > deadline - 5:
                break
            if (current >> (c - 1)) & 1:
                continue
            bit = 1 << (c - 1)
            trial = current | bit
            if _is_valid_mask(trial, bad_masks):
                current = trial
        
        if bin(current).count('1') > best_size:
            best_size = bin(current).count('1')
            best_mask = current
    
    # Strategy 4: Local search - try swaps
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removing one element and adding another
            for remove_c in range(1, 25):
                if time.time() > deadline - 10:
                    break
                if not ((best_mask >> (remove_c - 1)) & 1):
                    continue
                remove_bit = 1 << (remove_c - 1)
                base = best_mask & ~remove_bit
                for add_c in range(1, 25):
                    if time.time() > deadline - 10:
                        break
                    if add_c == remove_c:
                        continue
                    if (base >> (add_c - 1)) & 1:
                        continue
                    add_bit = 1 << (add_c - 1)
                    trial = base | add_bit
                    if _is_valid_mask(trial, bad_masks):
                        if bin(trial).count('1') > bin(best_mask).count('1'):
                            best_mask = trial
                            best_size = bin(trial).count('1')
                            improved = True
                        elif bin(trial).count('1') == bin(best_mask).count('1'):
                            # Same size, but might enable further improvements
                            # Accept if it leads to a different configuration
                            pass
    
    # Strategy 5: Try to find larger sets by exhaustive search for sizes above best
    if time.time() < deadline - 20:
        import itertools
        # Try sizes from 24 down to best_size + 1
        for k in range(24, best_size, -1):
            if time.time() > deadline - 5:
                break
            found = False
            # Generate combinations of size k
            # This is only feasible for k close to 24
            if k >= 22:
                for comb in itertools.combinations(range(1, 25), k):
                    if time.time() > deadline - 5:
                        found = True
                        break
                    mask = 0
                    for x in comb:
                        mask |= (1 << (x - 1))
                    if _is_valid_mask(mask, bad_masks):
                        best_mask = mask
                        best_size = k
                        found = True
                        break
            if found:
                break
    
    # Convert best_mask to list
    result = []
    for x in range(1, 25):
        if (best_mask >> (x - 1)) & 1:
            result.append(x)
    
    # Ensure non-empty
    if not result:
        result = [24]
    
    return sorted(result)
