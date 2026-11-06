import time
import random
import numpy as np

N = 512

def _has_4ap(bitset):
    # Returns True if any nonconstant 4-AP exists in the set represented by bitset.
    # We iterate over all possible 4-APs (a, a+d, a+2d, a+3d) within [1, N] and check
    # if all four are in the set.
    # To make this fast, we can precompute all 4-APs.
    # But for dynamic checking during greedy, we need to check if adding x creates a new 4-AP.
    # A new 4-AP involving x can have x as a, a+d, a+2d, or a+3d.
    # Case 1: x is the first term. Need x+d, x+2d, x+3d in set.
    # Case 2: x is the second term. Need x-d, x+d, x+2d in set.
    # Case 3: x is the third term. Need x-2d, x-d, x+d in set.
    # Case 4: x is the fourth term. Need x-3d, x-2d, x-d in set.
    
    # We will implement a helper that checks if adding x to a given bitset creates a 4-AP.
    # This is much faster than checking the whole set each time.
    
    pass

def _creates_4ap(x, bitset):
    # Check if adding x to the set (represented by bitset) creates a 4-AP.
    # x is 1-indexed. Bitset is 0-indexed or 1-indexed? Let's use 1-indexed for clarity, 
    # but numpy arrays are 0-indexed. Let's use 0-indexed internally: index = val - 1.
    # So x corresponds to index x-1.
    
    # Case 1: x is the first term (a).
    # a, a+d, a+2d, a+3d. x = a.
    # We need x+d, x+2d, x+3d to be in the set.
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (bitset[x-1+d] and bitset[x-1+2*d] and bitset[x-1+3*d]):
            return True
            
    # Case 2: x is the second term (a+d).
    # a, a+d, a+2d, a+3d. x = a+d => a = x-d.
    # We need x-d, x+d, x+2d to be in the set.
    max_d = (N - x) // 2
    for d in range(1, max_d + 1):
        if (x-d >= 1) and (bitset[x-2-d] and bitset[x-1+d] and bitset[x-1+2*d]):
            return True
            
    # Case 3: x is the third term (a+2d).
    # a, a+d, a+2d, a+3d. x = a+2d => a = x-2d.
    # We need x-2d, x-d, x+d to be in the set.
    max_d = (N - x) // 1
    for d in range(1, max_d + 1):
        if (x-2*d >= 1) and (bitset[x-1-2*d] and bitset[x-1-d] and bitset[x-1+d]):
            return True
            
    # Case 4: x is the fourth term (a+3d).
    # a, a+d, a+2d, a+3d. x = a+3d => a = x-3d.
    # We need x-3d, x-2d, x-d to be in the set.
    max_d = (N - x) // 0 # No, max_d is determined by a >= 1 => x-3d >= 1 => d <= (x-1)/3
    for d in range(1, (x-1)//3 + 1):
        if (bitset[x-1-3*d] and bitset[x-1-2*d] and bitset[x-1-d]):
            return True
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    best_set = set()
    best_bitset = np.zeros(N, dtype=bool)
    
    # Initial deterministic run (ascending) as a baseline
    current_bitset = np.zeros(N, dtype=bool)
    for x in range(1, N + 1):
        if not _creates_4ap(x, current_bitset):
            current_bitset[x-1] = True
    best_bitset = current_bitset.copy()
    
    # Randomized greedy restarts
    # We try different random orderings.
    # To speed up, we can also try "reverse" or "random shuffle".
    
    while time.time() < deadline:
        # Generate a random permutation of [1, N]
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        current_bitset = np.zeros(N, dtype=bool)
        
        for x in order:
            if time.time() > deadline:
                break
            if not _creates_4ap(x, current_bitset):
                current_bitset[x-1] = True
        
        if current_bitset.sum() > best_bitset.sum():
            best_bitset = current_bitset.copy()
            
    # Convert best_bitset to a sorted list
    result = [i+1 for i in range(N) if best_bitset[i]]
    
    # Ensure nonempty (should always be true)
    if not result:
        result = [1]
        
    return result

# To make _creates_4ap faster, we can precompute the indices to check for each x.
# But since N=512 is small, the loops are short.
# Let's optimize _creates_4ap by using numpy bitwise operations where possible?
# Actually, python loops for small ranges are okay.
# Let's try to make it even faster by precomputing the d-ranges.

def _precompute_checks():
    checks = {}
    for x in range(1, N + 1):
        # List of indices to check for each case
        # Case 1: x is 1st. Check x+d, x+2d, x+3d.
        # Case 2: x is 2nd. Check x-d, x+d, x+2d.
        # Case 3: x is 3rd. Check x-2d, x-d, x+d.
        # Case 4: x is 4th. Check x-3d, x-2d, x-d.
        
        # We store tuples of indices (0-based) that must ALL be True for a conflict.
        # If any tuple is all True, adding x creates a 4-AP.
        
        conflicts = []
        
        # Case 1
        for d in range(1, (N - x) // 3 + 1):
            conflicts.append((x-1+d, x-1+2*d, x-1+3*d))
            
        # Case 2
        for d in range(1, (N - x) // 2 + 1):
            if x - d >= 1:
                conflicts.append((x-2-d, x-1+d, x-1+2*d))
                
        # Case 3
        for d in range(1, N - x + 1):
            if x - 2*d >= 1:
                conflicts.append((x-1-2*d, x-1-d, x-1+d))
                
        # Case 4
        for d in range(1, (x-1)//3 + 1):
            conflicts.append((x-1-3*d, x-1-2*d, x-1-d))
            
        checks[x] = conflicts
    return checks

CHECKS = _precompute_checks()

def _creates_4ap_fast(x, bitset):
    for idx_tuple in CHECKS[x]:
        if bitset[idx_tuple[0]] and bitset[idx_tuple[1]] and bitset[idx_tuple[2]]:
            return True
    return False

# Replace the original _creates_4ap with the fast version
_creates_4ap = _creates_4ap_fast

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    best_bitset = np.zeros(N, dtype=bool)
    
    # Initial deterministic run (ascending) as a baseline
    current_bitset = np.zeros(N, dtype=bool)
    for x in range(1, N + 1):
        if not _creates_4ap(x, current_bitset):
            current_bitset[x-1] = True
    best_bitset = current_bitset.copy()
    
    # Randomized greedy restarts
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        current_bitset = np.zeros(N, dtype=bool)
        
        for x in order:
            if time.time() > deadline:
                break
            if not _creates_4ap(x, current_bitset):
                current_bitset[x-1] = True
        
        if current_bitset.sum() > best_bitset.sum():
            best_bitset = current_bitset.copy()
            
    result = [i+1 for i in range(N) if best_bitset[i]]
    
    if not result:
        result = [1]
        
    return result
