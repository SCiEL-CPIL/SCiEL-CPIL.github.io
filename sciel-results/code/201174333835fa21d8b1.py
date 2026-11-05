import time
import itertools
import math

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute LCM
    L = 1
    for i in range(1, 25):
        L = _lcm(L, i)
        
    # Precompute values L//i
    vals = [0] * 25
    for i in range(1, 25):
        vals[i] = L // i

    # Find all violations (x, mask_A)
    # x is the element whose reciprocal is the sum
    # mask_A is the subset of OTHER elements that sum to 1/x
    violations = []
    
    # Helper to find all subsets of available elements summing to target
    def find_bad_masks(x, target, available_elements):
        # available_elements is a list of (val, elem_idx)
        # We want subsets of available_elements summing to target
        n = len(available_elements)
        if n == 0:
            return []
        
        # Sort by value descending for pruning
        available_elements.sort(key=lambda p: p[0], reverse=True)
        
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + available_elements[i][0]
            
        masks = []
        
        def dfs(start_idx, current_sum, current_mask):
            if current_sum == target:
                if current_mask != 0:
                    masks.append(current_mask)
                return
            
            if current_sum > target:
                return
            
            if start_idx >= n:
                return
            
            if current_sum + suffix_sums[start_idx] < target:
                return
            
            for i in range(start_idx, n):
                val, elem_idx = available_elements[i]
                new_sum = current_sum + val
                if new_sum > target:
                    continue
                dfs(i + 1, new_sum, current_mask | (1 << (elem_idx - 1)))
        
        dfs(0, 0, 0)
        return masks

    for x in range(1, 25):
        target = vals[x]
        # Elements available are 1..24 excluding x
        elems = []
        for y in range(1, 25):
            if y != x:
                elems.append((vals[y], y))
        
        bad_masks = find_bad_masks(x, target, elems)
        for bm in bad_masks:
            violations.append((x, bm))
            
    # Optimization: Remove redundant violations?
    # If (x, A1) and (x, A2) are violations, and A1 is a subset of A2, then A1 is "stronger"?
    # No, if A1 is a subset of A2, and we select A2, we also select A1.
    # So if A1 is a bad mask, any superset of A1 is also "bad" in the sense that it contains the bad subset.
    # So we only need to keep the *minimal* bad masks?
    # If we select a set M, it is invalid if it contains *any* bad mask A.
    # If A1 subset A2, and M contains A2, it contains A1.
    # So if we check for A1, we don't need to check for A2?
    # Yes! We only need to check the *minimal* bad masks (by inclusion).
    # Let's filter violations to only keep minimal masks for each x.
    
    minimal_violations = []
    # Group by x
    from collections import defaultdict
    violations_by_x = defaultdict(list)
    for x, bm in violations:
        violations_by_x[x].append(bm)
        
    for x, masks in violations_by_x.items():
        # Sort by bit count
        masks.sort(key=lambda m: bin(m).count('1'))
        minimal = []
        for m in masks:
            # Check if m is a superset of any already kept minimal mask
            is_superset = False
            for km in minimal:
                if (m & km) == km:
                    is_superset = True
                    break
            if not is_superset:
                minimal.append(m)
        for m in minimal:
            minimal_violations.append((x, m))
            
    violations = minimal_violations
    
    # Backtracking search for max valid set
    # We want to maximize the number of elements.
    # Order: 24 down to 1 (Large elements first, easier to include)
    # State: idx (current element index 24..1), current_mask, current_size
    
    best_mask = 0
    best_size = 0
    
    # Precompute validity check function
    def is_valid(mask):
        for x, bm in violations:
            if (mask & (1 << (x - 1))) and (mask & bm) == bm:
                return False
        return True

    def dfs(idx, current_mask, current_size):
        nonlocal best_mask, best_size
        
        if time.time() > deadline - 2:
            return
            
        # Prune
        if current_size + (24 - idx) <= best_size:
            return
            
        if idx == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Option 1: Exclude idx
        dfs(idx - 1, current_mask, current_size)
        
        # Option 2: Include idx
        new_mask = current_mask | (1 << (idx - 1))
        if is_valid(new_mask):
            dfs(idx - 1, new_mask, current_size + 1)

    # Start with 24 down to 1
    # idx represents the element number (1-24)
    dfs(24, 0, 0)
    
    # Extract result
    result = [x for x in range(1, 25) if (best_mask >> (x - 1)) & 1]
    
    # Fallback if search failed or returned empty
    if not result:
        # Greedy fallback
        current_mask = 0
        for x in range(24, 0, -1):
            trial = current_mask | (1 << (x - 1))
            if is_valid(trial):
                current_mask = trial
        result = [x for x in range(1, 25) if (current_mask >> (x - 1)) & 1]
        
    if not result:
        result = [24]
        
    return result
