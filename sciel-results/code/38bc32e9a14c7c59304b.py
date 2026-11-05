import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, keep 5s buffer
    
    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Generate all divisors of target that are > 12
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    divs = sorted(set(divisors))
    n_divs = len(divs)
    
    # We need at most 7 factors (since 13*14*...*20 > 12!)
    # Search for minimum maximum factor using DFS with pruning
    
    best_solution = None
    best_max = target  # fallback: single factor
    
    # For pruning: if we've picked 'count' factors so far and need 'remaining' more,
    # the product of the remaining 'remaining' smallest available divisors must be <= rem
    # and the product of the remaining 'remaining' largest available divisors must be >= rem
    
    def dfs(rem, start_idx, max_allowed_idx, current_factors, depth):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return
        
        if rem == 1:
            if 1 <= len(current_factors) <= 12:
                m = max(current_factors)
                if m < best_max:
                    best_max = m
                    best_solution = list(current_factors)
            return
        
        # Pruning: if remaining product is too large for the remaining slots
        remaining_slots = 12 - len(current_factors)
        if remaining_slots <= 0:
            return
        
        # If rem itself is a valid factor and we can use it
        if rem > 12 and rem <= divs[max_allowed_idx]:
            # Check if rem is in our divisor list and >= divs[start_idx]
            # Find index of rem in divs
            import bisect
            idx = bisect.bisect_left(divs, rem, start_idx, max_allowed_idx + 1)
            if idx <= max_allowed_idx and divs[idx] == rem:
                current_factors.append(rem)
                m = max(current_factors)
                if m < best_max:
                    best_max = m
                    best_solution = list(current_factors)
                current_factors.pop()
        
        # Try each divisor from start_idx to max_allowed_idx
        for i in range(start_idx, max_allowed_idx + 1):
            d = divs[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            # Pruning: the remaining product rem//d must be achievable with 
            # at most (remaining_slots - 1) more factors, all > d and <= divs[max_allowed_idx]
            new_rem = rem // d
            new_remaining_slots = remaining_slots - 1
            
            if new_remaining_slots <= 0:
                # We must have new_rem == 1, already handled above
                continue
            
            # Lower bound: product of new_remaining_slots smallest divisors after i
            # Upper bound: product of new_remaining_slots largest divisors up to max_allowed_idx
            # Simple pruning: if new_rem > 1 and new_remaining_slots == 1, new_rem must be a divisor
            if new_remaining_slots == 1:
                # new_rem must be a divisor > d and <= divs[max_allowed_idx]
                if new_rem > d and new_rem <= divs[max_allowed_idx] and new_rem > 12:
                    import bisect
                    idx2 = bisect.bisect_left(divs, new_rem, i + 1, max_allowed_idx + 1)
                    if idx2 <= max_allowed_idx and divs[idx2] == new_rem:
                        current_factors.append(d)
                        current_factors.append(new_rem)
                        m = max(current_factors)
                        if m < best_max:
                            best_max = m
                            best_solution = list(current_factors)
                        current_factors.pop()
                        current_factors.pop()
                continue
            
            # For more complex pruning, just recurse
            current_factors.append(d)
            dfs(new_rem, i + 1, max_allowed_idx, current_factors, depth + 1)
            current_factors.pop()
    
    # Search with increasing max_allowed factor to find the minimum possible max
    # Start from a reasonable lower bound
    # The minimum possible max for k factors is at least (target)^(1/k)
    # For k=7: ~16.2, so try max from 17 upward
    
    # Try different upper bounds on the maximum factor
    for max_val in range(17, 200):
        if time.time() > deadline:
            break
        
        # Find the index of the largest divisor <= max_val
        import bisect
        max_idx = bisect.bisect_right(divs, max_val) - 1
        if max_idx < 0:
            continue
        
        # DFS to find if we can form target using divisors in [13, max_val]
        dfs(target, 0, max_idx, [], 0)
        
        if best_solution is not None and best_max <= max_val:
            break
    
    if best_solution is not None:
        return sorted(best_solution)
    
    # Fallback: return a valid solution
    return [target]
