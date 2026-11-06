import math
import time

def construct():
    target = math.factorial(12)
    deadline = time.time() + 85.0
    
    # 1. Get all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12
    candidates = sorted([d for d in divisors if d > 12])
    
    # Precompute index for fast lookup? Not strictly necessary for 792 items.
    # But we need to check if a number is a candidate and find its index.
    # For the pruning "next factor > d", we just iterate from start_idx.
    
    best_max = target
    best_factors = [target]
    
    # Helper to check if remaining can be formed by factors > last_factor and <= limit
    # We do this by checking if there exists a divisor of 'remaining' in candidates 
    # that is > last_factor and <= limit.
    # If no such divisor exists, we prune.
    
    def dfs(remaining, start_idx, current_factors):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
            
        if remaining == 1:
            current_max = max(current_factors)
            if current_max < best_max:
                best_max = current_max
                best_factors = current_factors[:]
            return
            
        # If we have used 12 factors and remaining != 1, fail
        if len(current_factors) >= 12:
            return
            
        # Iterate through candidates starting from start_idx
        for i in range(start_idx, len(candidates)):
            d = candidates[i]
            
            # If d >= best_max, we can't improve best_max with this or any larger d
            if d >= best_max:
                break
                
            if remaining % d != 0:
                continue
                
            new_remaining = remaining // d
            
            # Pruning: Can new_remaining be formed by factors > d and < best_max?
            # If new_remaining == 1, handled above.
            # If new_remaining > 1:
            # It must have at least one divisor in candidates that is > d and < best_max.
            # Let's find the smallest such divisor.
            # Actually, we just recurse. The recursion will handle it.
            # But we can prune if no divisor of new_remaining exists in candidates[i+1:] that is < best_max.
            
            # Quick check: if new_remaining is prime, it must be a candidate > d and < best_max.
            # We can check primality or just rely on the next step.
            # Given the small size, direct recursion is likely fine.
            
            # Additional prune: 
            # The smallest possible factor for the next step is candidates[i+1] (if exists).
            # If new_remaining < candidates[i+1] (and i+1 < len), then new_remaining cannot be formed 
            # by any single factor > d.
            # Can it be formed by multiple?
            # Min product of 2 factors > d is candidates[i+1] * candidates[i+2].
            # If new_remaining < candidates[i+1] * candidates[i+2], it can't be formed by 2 or more factors.
            # So if new_remaining < candidates[i+1] (for single) and new_remaining < candidates[i+1]*candidates[i+2] (for double),
            # it must be a single factor.
            # So if new_remaining is not a candidate > d and < best_max, and it's too small for 2 factors, prune.
            
            if i + 1 < len(candidates):
                min_next = candidates[i+1]
                if new_remaining < min_next:
                    # Must be single factor
                    if new_remaining != 1: # Already handled
                        # Check if new_remaining is a candidate > d and < best_max
                        # Since candidates is sorted, if new_remaining is in candidates, 
                        # its index must be > i.
                        # We can check if new_remaining in candidates[i+1:] and < best_max
                        # But we also need it to divide new_remaining (which is trivially true if it equals it).
                        # So we just need new_remaining to be in candidates and < best_max.
                        # And since we are in the loop for d, and new_remaining < min_next, 
                        # if new_remaining is a candidate, it must be at index > i.
                        # But wait, if new_remaining < min_next, it cannot be in candidates[i+1:].
                        # So it's impossible.
                        continue
                else:
                    # Could be single or multiple.
                    # If it's single, it must be in candidates[i+1:] and < best_max.
                    # If it's multiple, it must be divisible by some candidate in candidates[i+1:] < best_max.
                    # We let the recursion handle this.
                    pass
            else:
                # No more candidates
                if new_remaining != 1:
                    continue

            current_factors.append(d)
            dfs(new_remaining, i + 1, current_factors)
            current_factors.pop()

    dfs(target, 0, [])
    
    return best_factors
