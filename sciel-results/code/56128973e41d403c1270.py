import time
import random
import numpy as np

def _fast_valid_check(S):
    """
    Checks if the set S is valid:
    1. All elements distinct, in 1..256.
    2. No two distinct non-empty subset sums divide each other.
    """
    n = len(S)
    if n == 0:
        return True
    if len(set(S)) != n:
        return False
    if any(x < 1 or x > 256 for x in S):
        return False
    
    # Generate all non-empty subset sums
    # Using DP / iterative set expansion
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        new_sums.add(x)
        sums.update(new_sums)
    
    if not sums:
        return True
        
    max_sum = max(sums)
    # Boolean array for O(1) lookup
    present = [False] * (max_sum + 1)
    for s in sums:
        present[s] = True
        
    # Check divisibility
    # For each sum s, check if any multiple k*s (k>=2) is in sums
    # To avoid double checking, we can just check all s.
    # Optimization: if s > max_sum // 2, it can't divide any other sum in the set (since 2s > max_sum)
    # So we only need to check s <= max_sum // 2
    
    for s in sums:
        if s > max_sum // 2:
            continue
        # Check multiples
        # We can iterate multiples or check if any sum is divisible by s
        # Iterating multiples is faster if s is small?
        # Number of multiples is max_sum // s
        # If s is small, many multiples. But s is a sum, so s >= min(S).
        # If min(S) is large, this is fast.
        
        # Alternative: iterate through all sums and check mod?
        # That's O(K^2).
        # Iterating multiples is O(sum(max_sum/s)).
        
        multiple = 2 * s
        while multiple <= max_sum:
            if present[multiple]:
                return False
            multiple += s
            
    return True

def construct():
    start = time.time()
    deadline = start + 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    # Strategy:
    # 1. Try descending order (256 down to 1). This tends to find valid sets because large numbers have fewer multiples.
    # 2. Try random permutations, biased towards larger numbers.
    
    # Restart 0: Pure Descending
    candidates = list(range(256, 0, -1))
    current = []
    for x in candidates:
        if time.time() > deadline:
            break
        trial = current + [x]
        if _fast_valid_check(trial):
            current = trial
            if len(current) > best_size:
                best_size = len(current)
                best_set = list(current)
                
    # If we hit deadline, return best
    if time.time() > deadline:
        return [int(x) for x in best_set] if best_set else [11, 13, 17]

    # Subsequent restarts: Randomized
    # We want to explore different combinations.
    # Bias: Larger numbers are easier to place.
    
    for trial in range(1, 1000):
        if time.time() > deadline - 1.0:
            break
            
        # Generate a candidate list
        # Mix of high and low numbers
        # Let's shuffle, but maybe weight higher numbers?
        # Simple shuffle is okay, but descending worked well.
        # Let's try: Shuffle, but keep the top 50% in descending order?
        # Or just shuffle range(1, 257)
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current = []
        for x in candidates:
            if time.time() > deadline - 0.5:
                break
            trial_set = current + [x]
            if _fast_valid_check(trial_set):
                current = trial_set
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
                    
    # Final check
    if best_set:
        # Ensure types are int
        return [int(x) for x in best_set]
    else:
        return [11, 13, 17]
