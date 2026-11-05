"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _check_x_is_mean_of_subset(x, S):
    """
    Check if x is the mean of any subset of at least 2 elements in S.
    Returns True if violation found.
    """
    n = len(S)
    if n < 2:
        return False
    
    zs = [y - x for y in S]
    
    # Calculate max absolute sum for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False
        
    offset = max_sum
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s-offset is achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            
            dp[k] |= (shifted & mask)
            
            # Early exit: if sum 0 is achievable with k >= 2 elements
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
                
    return False

def _check_existing_violation(x, S):
    """
    Check if adding x to S creates any violation where an element a in S
    is the mean of a subset containing x and at least one other element from S.
    
    For each a in S, we need to check: is a the mean of {x} ∪ T where T ⊆ S\{a}, |T| >= 1?
    This means: x + sum(T) = (|T|+1)*a
    sum(T) = (|T|+1)*a - x
    Let w_t = t - a for t in S\{a}.
    sum_{t in T} w_t = a - x
    
    We need to know if there is a non-empty subset T of S\{a} such that sum of (t-a) is a-x.
    """
    n = len(S)
    if n < 2:
        return False
    
    for a in S:
        target_val = a - x
        # Elements in S other than a
        U = [t for t in S if t != a]
        if not U:
            continue
            
        ws = [t - a for t in U]
        max_abs = sum(abs(w) for w in ws)
        if max_abs == 0:
            continue
        
        offset = max_abs
        mask = (1 << (2 * max_abs + 1)) - 1
        
        # dp[k] = bitset for sums achievable with exactly k elements
        dp = [0] * (len(U) + 1)
        dp[0] = 1 << offset
        
        for w in ws:
            for k in range(len(U), 0, -1):
                if dp[k-1] == 0:
                    continue
                if w >= 0:
                    shifted = dp[k-1] << w
                else:
                    shifted = dp[k-1] >> (-w)
                dp[k] |= (shifted & mask)
        
        # Check if target_val is achievable with any k >= 1
        # target_val corresponds to bit (offset + target_val)
        if 0 <= offset + target_val <= 2 * max_abs:
            for k in range(1, len(U) + 1):
                if dp[k] & (1 << (offset + target_val)):
                    return True
    
    return False

def _quick_pair_check(x, S):
    """
    Quick check for 2-element subsets involving x.
    1. Is x the mean of a pair in S? (2x = a+b)
    2. Is any a in S the mean of pair (x, b) for b in S, b != a? (2a = x+b => b = 2a-x)
    """
    # 1. Is x mean of a pair in S?
    S_set = set(S)
    for a in S:
        b = 2 * x - a
        if b in S_set and b != a:
            return True
    
    # 2. Is any a in S mean of pair (x, b)?
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a:
            return True
    
    return False

def _full_check(C):
    """
    Full verification: check if any element in C is the mean of a subset of 
    at least 2 other elements.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        if _check_x_is_mean_of_subset(a, others):
            return True
    return False

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    rng = random.Random(42)
    
    # We will run multiple greedy passes with different orders
    # and then try local search improvements.
    
    num_greedy_passes = 50
    
    for _ in range(num_greedy_passes):
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        # Create a random order, sometimes biased
        order = candidates[:]
        # Mix of pure random and structured
        r = rng.random()
        if r < 0.2:
            pass # Already random
        elif r < 0.4:
            order.sort()
        elif r < 0.6:
            order.sort(reverse=True)
        else:
            # Bias towards extremes
            # Sample from 1-100 and 400-512 more often
            # Simple way: shuffle, but move some to front/back?
            # Just shuffle for now, it's robust
            pass
        rng.shuffle(order)
        
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 10:
                break
                
            if len(selected) >= 48:
                break
                
            # Quick check
            if _quick_pair_check(x, selected):
                continue
            
            # Check if x is mean of subset of selected
            if _check_x_is_mean_of_subset(x, selected):
                continue
            
            # Check if any existing element becomes mean of subset with x
            if _check_existing_violation(x, selected):
                continue
            
            selected.append(x)
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        # Local Search Phase for this specific selected set
        # Try to improve it by swapping elements
        # Budget: 2 seconds per pass? Or total?
        # Let's say we spend up to 5 seconds on local search for the best set found so far
        # Actually, let's do local search on the current `selected` if it's close to best.
        
    # Final Local Search on the global best_set
    if best_set:
        # Try to add more elements if < 48
        # Try to swap to increase size? No, size is fixed or increasing.
        # If we are at 48, we are done.
        # If we are < 48, we might be stuck.
        
        # Try to fill remaining slots
        if len(best_set) < 48:
            # Try all candidates not in best_set
            current_set = set(best_set)
            remaining_candidates = [c for c in candidates if c not in current_set]
            # Shuffle them
            rng.shuffle(remaining_candidates)
            
            for x in remaining_candidates:
                if time.time() - start_time > _DEADLINE - 2:
                    break
                if len(best_set) >= 48:
                    break
                    
                # Check if x can be added
                if _quick_pair_check(x, best_set):
                    continue
                if _check_x_is_mean_of_subset(x, best_set):
                    continue
                if _check_existing_violation(x, best_set):
                    continue
                
                best_set.append(x)
                
        # Try local search: remove one, add two?
        # Or just swap?
        # Swapping doesn't increase size.
        # We want to increase size.
        # If we are stuck at size K < 48, we need to find a way to fit more.
        # This implies the current set is "too dense" or "badly chosen".
        # A restart with a different random seed/order is better than local search for size increase.
        # So the multiple greedy passes above handle this.
        
        # Final verification
        if _full_check(best_set):
             # Fallback
             return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
             return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
