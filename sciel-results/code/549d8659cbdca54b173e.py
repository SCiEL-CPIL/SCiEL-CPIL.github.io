import time
import random

def _is_valid(s):
    """
    Check if the set s has cardinality-separated subset sums.
    Condition: For all k, sum of largest k < sum of smallest k+1.
    """
    n = len(s)
    if n < 2:
        return True
    
    s_sorted = sorted(s)
    
    # Precompute prefix sums
    # prefix[i] = sum of s_sorted[0..i-1]
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + s_sorted[i]
        
    # Check condition for k = 1 to n-2
    # X_k = sum of largest k = sum(s_sorted[n-k:n]) = prefix[n] - prefix[n-k]
    # M_{k+1} = sum of smallest k+1 = sum(s_sorted[0:k+1]) = prefix[k+1]
    
    for k in range(1, n - 1):
        # k is the cardinality of the subset for X
        # We compare X_k with M_{k+1}
        # Note: The problem says "Whenever two subsets have different cardinalities their sums must differ".
        # This is equivalent to ranges of sums for different cardinalities not overlapping.
        # Since sums are integers, if min(sum_{k+1}) > max(sum_k), they are disjoint.
        # min(sum_{k+1}) is sum of smallest k+1.
        # max(sum_k) is sum of largest k.
        
        x_k = prefix[n] - prefix[n - k]
        m_k1 = prefix[k + 1]
        
        if x_k >= m_k1:
            return False
            
    return True

def construct():
    best_set = []
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Try multiple randomized greedy runs
    # We can also try different initial seeds or biases
    for _ in range(10000): # Enough iterations if fast
        if time.time() > deadline:
            break
            
        current_set = []
        # Shuffle candidates to explore different orders
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for c in candidates:
            # Optimization: if current_set is already larger than best, we might want to check more carefully?
            # No, just add if valid.
            # Check validity of current_set + [c]
            # To speed up, we can skip if len(current_set) + 1 <= len(best_set) and we are not exploring for improvement?
            # But we need to find a valid set to add.
            
            # Fast check:
            # We can just call _is_valid. It's O(n^2).
            # With n ~ 20-30, this is very fast.
            
            # Heuristic: Smaller numbers are "harder" to satisfy with large numbers?
            # Actually, let's just try.
            
            new_set = current_set + [c]
            if _is_valid(new_set):
                current_set = new_set
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    # Ensure return type is list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
