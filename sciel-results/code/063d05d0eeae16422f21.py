import time
import random
import sys

def _is_valid_set(S):
    """
    Check if S is a valid set.
    Returns True if valid.
    """
    if len(S) < 3:
        return True
        
    S = list(S)
    N = len(S)
    
    # For each element, check if it's the mean of any subset of others
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        max_sum = sum(others)
        
        # Initialize dp
        dp = [0] * (n_others + 1)
        dp[0] = 1  # bit 0 set
        
        for val in others:
            # Update dp in reverse to avoid reusing same element
            for k in range(n_others - 1, -1, -1):
                if dp[k] == 0:
                    continue
                shifted = dp[k] << val
                dp[k+1] |= shifted
                
        # Check if any dp[k] for k >= 2 has a bit set at position k*a
        for k in range(2, n_others + 1):
            if dp[k] == 0:
                continue
            target_sum = k * a
            if target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
            else:
                break # Since k increases, target_sum increases. If it's too big for this k, it will be too big for larger k?
                # Wait. max_sum is fixed. k*a increases with k.
                # So if k*a > max_sum, then for any j > k, j*a > k*a > max_sum.
                # So we can break.
                pass
                
    return True

def _fast_pair_check(S, x):
    """
    Quick check for pair violations involving x and S.
    Returns True if a pair violation is found.
    """
    S_set = set(S)
    
    # 1. x is mean of a, b in S
    # 2x = a + b
    for a in S:
        b = 2 * x - a
        if b in S_set and b != a:
            return True
            
    # 2. Some a in S is mean of x, b in S
    # 2a = x + b => b = 2a - x
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a and b != x:
            return True
            
    return False

def _greedy_sidon():
    """
    Construct a large Sidon set (B2 sequence) in [1, 512].
    A Sidon set has all pairwise sums distinct.
    This implies no element is the mean of two others.
    """
    S = []
    used_sums = set()
    for x in range(1, 513):
        valid = True
        for y in S:
            s = x + y
            if s in used_sums:
                valid = False
                break
        if valid:
            # Add x and update used_sums
            for y in S:
                used_sums.add(x + y)
            S.append(x)
    return S

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = []
    
    # Strategy 1: Start with a large Sidon set, then extend greedily
    # The Sidon set guarantees no k=2 violations.
    # We then try to add more elements, checking k>=3 violations.
    
    base_sidon = _greedy_sidon()
    
    # Try to extend the Sidon set
    # We will use a greedy approach on the remaining candidates.
    # Order: try to add numbers that are "far" from existing ones?
    # Or just iterate 1..512?
    
    # Let's try a few orders for the extension phase.
    orders = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        list(range(1, 513, 2)), # Odds first?
        list(range(2, 513, 2)), # Evens first?
    ]
    
    # Add some random orders
    rng = random.Random(42)
    for _ in range(20):
        rand_order = list(range(1, 513))
        rng.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
            
        current_set = base_sidon[:]
        current_set_set = set(current_set)
        
        # We need to check if the base_sidon itself is valid for k>=3.
        # It is a Sidon set, so k=2 is safe.
        # But it might have k>=3 violations.
        if not _is_valid_set(current_set):
            # If base is invalid, we can't use it as a base for extension in this order?
            # Or we can prune it?
            # Let's just skip this base if it's invalid?
            # Actually, _is_valid_set is expensive.
            # Let's assume base_sidon is likely valid or close.
            # If it's invalid, we should probably not start with it.
            # But checking it costs time.
            # Let's check it once.
            pass # We'll rely on the extension loop to find a valid subset if we start empty?
            # No, starting with a large set is better.
            # Let's check validity of base_sidon once outside.
            
    # Check base_sidon validity once
    if _is_valid_set(base_sidon):
        best_set = base_sidon[:]
    else:
        # Fallback to smaller valid set from base
        # Greedily build valid set from base_sidon
        best_set = []
        for x in base_sidon:
            if time.time() - start_time > deadline - 5:
                break
            if _is_valid_set(best_set + [x]):
                best_set.append(x)

    # Now try to extend best_set
    # We will iterate through orders and try to add elements to best_set
    # Note: We should copy best_set for each order attempt to avoid contamination
    # Or just keep the global best and try to improve it.
    
    # Let's try to improve the current best_set
    # We'll use a local search or greedy extension.
    
    # To save time, we'll only try a few extension attempts.
    
    for order in orders:
        if time.time() - start_time > deadline - 2:
            break
            
        # Start from the current best_set
        current_set = best_set[:]
        current_set_set = set(current_set)
        
        for x in order:
            if time.time() - start_time > deadline - 1:
                break
                
            if len(current_set) >= 48:
                break
                
            if x in current_set_set:
                continue
                
            # Fast pair check
            if _fast_pair_check(current_set, x):
                continue
                
            # Full check
            test_set = current_set + [x]
            if _is_valid_set(test_set):
                current_set.append(x)
                current_set_set.add(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    return best_set
