import random
import time

def _get_subset_sums(A):
    """
    Returns an integer where bit i is 1 if sum i is possible using a subset of A.
    """
    bits = 1
    for v in A:
        bits |= bits << v
    return bits

def _is_valid_addition(S, x):
    """
    Checks if x can be added to set S without violating the mean condition.
    S is a list of distinct integers.
    """
    n = len(S)
    
    # Case 1: x is the mean of a subset of S
    if n >= 2:
        bits_S = _get_subset_sums(S)
        # Max sum is roughly n * 512
        for k in range(2, n + 1):
            target = k * x
            # Optimization: if target is too large, no subset can sum to it
            # Max possible sum of S is sum(S)
            if target > sum(S): 
                break
            if (bits_S >> target) & 1:
                return False

    # Case 2: Some a in S becomes the mean of a subset of S U {x}
    # The subset must contain x.
    # Let T be the subset. T = T' U {x}, T' subset of S\{a}.
    # sum(T) = |T| * a  =>  sum(T') + x = (|T'| + 1) * a
    # sum(T') = (|T'| + 1) * a - x
    # Let k = |T'| + 1. k >= 2 (since |T'| >= 1, as x!=a and T' non-empty? 
    # Wait, can T' be empty?
    # If T' is empty, sum(T')=0. Then x = 1*a => x=a. But x is distinct from a.
    # So T' must be non-empty. |T'| >= 1.
    # So k >= 2.
    # Max |T'| is n-1. So max k is n.
    
    S_list = list(S)
    for a in S_list:
        # Compute subset sums of S\{a}
        # We can filter S_list
        S_minus_a = [y for y in S_list if y != a]
        if len(S_minus_a) < 1:
            continue # No T' possible
            
        bits_S_minus_a = _get_subset_sums(S_minus_a)
        
        # Check for any k in [2, n]
        for k in range(2, n + 1):
            target = k * a - x
            if target < 0:
                break # Since a, x > 0 and k increases, target increases? 
                # No, k*a - x. If k=2, 2a-x. If k=3, 3a-x. 
                # If 2a-x < 0, 3a-x might be > 0 if a > x.
                # So we shouldn't break on < 0 unless we know target is monotonic and we've passed 0?
                # Actually, if target < 0, it's not a valid sum (sums are positive).
                # But larger k might yield positive target.
                # So we should continue, not break.
                continue
            
            # Max sum of S_minus_a is sum(S_minus_a)
            if target > sum(S_minus_a):
                continue
                
            if (bits_S_minus_a >> target) & 1:
                return False

    return True

def construct():
    best_set = []
    start_time = time.time()
    limit = 512
    max_size = 48
    
    # Use a fixed seed for reproducibility? No, random is better for search.
    # But to ensure we don't get unlucky, we run multiple restarts.
    
    while time.time() - start_time < 55:
        candidates = list(range(1, limit + 1))
        random.shuffle(candidates)
        S = []
        
        for x in candidates:
            if len(S) >= max_size:
                break
            if _is_valid_addition(S, x):
                S.append(x)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break
        
        # If we've been running for a while and haven't improved, 
        # we might try a different strategy?
        # For now, random restarts are sufficient.
        
    return best_set
