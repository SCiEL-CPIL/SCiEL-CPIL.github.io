import time
import random

def _is_valid_addition(S, x):
    """
    Check if adding x to set S results in a valid set.
    S is a list of integers.
    Returns True if valid, False otherwise.
    """
    n = len(S)
    if n == 0:
        return True
    
    # 1. Check if x is the mean of any subset of S
    # sum(T) = k * x for T subset of S, |T| = k >= 2
    max_sum = sum(S)
    # dp[k] is a bitset of sums of k elements from S
    # Max sum is max_sum. We only care up to n * x.
    max_target = n * x
    limit = min(max_sum, max_target)
    
    dp = [0] * (n + 1)
    dp[0] = 1
    # Mask to limit bits
    mask = (1 << (limit + 1)) - 1
    
    for elem in S:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << elem)
            dp[k] &= mask
            
    for k in range(2, n + 1):
        target = k * x
        if target <= limit:
            if (dp[k] >> target) & 1:
                return False
        # If target > limit, it's not possible since max sum is limit
        
    # 2. Check if any s in S becomes the mean of a subset including x
    # Subset T from S \ {s} such that x + sum(T) = s * (1 + |T|)
    # sum(T) = s * (1 + |T|) - x
    # Let |T| = t. t >= 0.
    # If t=0, sum(T)=0, x = s. But x not in S, so x != s.
    # So t >= 1.
    
    for s in S:
        # We need to check if there exists T subset of S \ {s} with |T|=t >= 1
        # such that sum(T) = s * (t + 1) - x
        
        # Build DP for S \ {s}
        others = [y for y in S if y != s]
        m_o = len(others)
        if m_o < 1:
            continue
            
        # Max target for t=m_o is s * (m_o + 1) - x
        max_t_target = s * (m_o + 1) - x
        if max_t_target < 0:
            continue
            
        min_t_target = s * 2 - x # t=1
        if min_t_target > max(sum(others)):
            continue
            
        limit_o = min(sum(others), max_t_target)
        mask_o = (1 << (limit_o + 1)) - 1
        
        dp_o = [0] * (m_o + 1)
        dp_o[0] = 1
        
        for elem in others:
            for k in range(m_o, 0, -1):
                dp_o[k] |= (dp_o[k-1] << elem)
                dp_o[k] &= mask_o
                
        for t in range(1, m_o + 1):
            target = s * (t + 1) - x
            if 0 <= target <= limit_o:
                if (dp_o[t] >> target) & 1:
                    return False
                    
    return True

def _is_valid_set(S):
    """
    Check if the set S is valid.
    """
    if len(S) <= 1:
        return True
        
    # Check all pairs? No, check all elements as means.
    # This is O(N * Cost_Addition) roughly.
    # We can just check each element x in S if it is the mean of S \ {x}.
    # This is equivalent to checking if S \ {x} has a subset T with sum(T) = k x.
    
    for x in S:
        others = [y for y in S if y != x]
        if not others:
            continue
            
        n_o = len(others)
        max_target = n_o * x
        limit = min(sum(others), max_target)
        
        dp = [0] * (n_o + 1)
        dp[0] = 1
        mask = (1 << (limit + 1)) - 1
        
        for elem in others:
            for k in range(n_o, 0, -1):
                dp[k] |= (dp[k-1] << elem)
                dp[k] &= mask
                
        for k in range(2, n_o + 1):
            target = k * x
            if target <= limit:
                if (dp[k] >> target) & 1:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy: Greedy with multiple orderings
    # Orderings: Random, Ascending, Descending, Primes, Powers of 2
    
    candidates = list(range(1, 513))
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        
        selected = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
                
            if _is_valid_addition(selected, x):
                selected.append(x)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            if best_size == 48:
                break
                
    # Local Search: Try to improve best_set
    if best_size < 48 and time.time() < deadline - 5:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any element
            for x in range(1, 513):
                if x in current:
                    continue
                if len(current) >= 48:
                    break
                if _is_valid_addition(current, x):
                    current.append(x)
                    improved = True
                    break
                    
            if not improved:
                # Try swaps
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    orig = current[i]
                    # Try replacing orig with x
                    # Check if current \ {orig} U {x} is valid
                    # This is expensive. 
                    # We can check if x is valid in current \ {orig}
                    # AND if removing orig doesn't leave current \ {orig} invalid?
                    # Removing elements preserves validity.
                    # So we just need to check if x is valid to add to current \ {orig}.
                    
                    reduced = [y for y in current if y != orig]
                    
                    # Check if reduced is still valid? It is, since it's a subset of a valid set.
                    
                    # Now check if x can be added to reduced
                    if _is_valid_addition(reduced, x):
                        # Check if the new set is valid?
                        # _is_valid_addition checks if adding x to reduced creates a violation.
                        # It checks:
                        # 1. x is mean of subset of reduced.
                        # 2. Any s in reduced is mean of subset of reduced U {x}.
                        # This covers all violations in the new set.
                        # So if it returns True, the new set is valid.
                        
                        current[i] = x
                        improved = True
                        break
                        
    return best_set[:48]
