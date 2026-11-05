import time
import random

def is_valid(S):
    """
    Check if no element in S is the arithmetic mean of any subset of at least two other elements in S.
    S is a list of distinct integers.
    """
    n = len(S)
    if n < 3:
        return True
    
    for a in S:
        others = [x for x in S if x != a]
        if len(others) < 2:
            continue
            
        # Compute all possible subset sums of 'others' using bitset
        bits = 1
        sum_others = 0
        for x in others:
            bits |= bits << x
            sum_others += x
            
        # Check if k * a is in the subset sums for any k >= 2
        # The smallest mean subset size is 2, so sum = 2a.
        # The largest possible sum is sum_others.
        target = a * 2
        while target <= sum_others:
            if (bits >> target) & 1:
                return False
            target += a
            
    return True

def construct():
    start_time = time.time()
    # Use 85 seconds to be safe against the 90s limit
    deadline = start_time + 85.0
    
    best_set = []
    
    # Initial deterministic greedy (ascending)
    for x in range(1, 513):
        if is_valid(best_set + [x]):
            best_set.append(x)
        if len(best_set) >= 48:
            return best_set[:48]
            
    # Randomized restarts
    # We try to find a set of size 48.
    # If we find one, we return immediately.
    
    # To improve chances, we can bias the random selection.
    # But pure random is a good baseline.
    
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        current_set = []
        candidates = list(range(1, 513))
        
        # Mix of random and slightly biased
        # Sometimes prefer larger numbers? Smaller?
        # Let's just shuffle.
        random.shuffle(candidates)
        
        for x in candidates:
            if len(current_set) >= 48:
                break
            # Check validity
            # Optimization: if current_set is large, is_valid is slower.
            # But we break at 48.
            if is_valid(current_set + [x]):
                current_set.append(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) >= 48:
                return best_set[:48]
                
    return best_set[:48]
