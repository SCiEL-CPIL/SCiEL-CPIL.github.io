import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    D = divisors
    
    # Precompute suffix sums of sorted divisors for pruning
    # suffix_sum[i] = sum(D[i:])
    suffix_sum = [0] * (len(D) + 1)
    for i in range(len(D) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + D[i]
        
    # Also prefix sums? No, we are picking distinct items.
    # To prune "can we reach target with remaining items", we use suffix_sum.
    # To prune "are we picking too small", we need to know max possible sum.
    # If we are at index i, and need to pick 'k' items, the max sum is sum of top k items from D[i:].
    # We can precompute top_k_sums or just use suffix_sum as an upper bound (sum of ALL remaining).
    # A tighter bound: if we need to pick k items, max sum is sum of first k items in D[i:].
    # Let's just use the simple bound: sum of remaining available >= target.
    # And: sum of largest k available >= target.
    
    def max_sum_k(start_idx, k):
        # Sum of k largest elements in D[start_idx:]
        # Since D is sorted desc, this is sum(D[start_idx : start_idx + k])
        # We can compute this quickly or precompute.
        # Given small N, simple sum is fine.
        s = 0
        for j in range(start_idx, min(len(D), start_idx + k)):
            s += D[j]
        return s

    def dfs(start_idx, remaining, k_left, current_set, deadline):
        if time.time() > deadline:
            return None
        if k_left == 0:
            if remaining == 0:
                return list(current_set)
            return None
        
        # Pruning 1: Not enough items left
        if len(D) - start_idx < k_left:
            return None
            
        # Pruning 2: Max possible sum with k_left items from start_idx is less than remaining
        if max_sum_k(start_idx, k_left) < remaining:
            return None
            
        # Pruning 3: Min possible sum?
        # The smallest k_left items from start_idx onwards.
        # Actually, we just iterate.
        
        # Iterate through possible next divisors
        # We can restrict the range of D[j]
        # D[j] must be <= remaining
        # Also, D[j] must be large enough so that remaining - D[j] can be formed by k_left-1 items from D[j+1:]
        # Min sum of k_left-1 items from D[j+1:] is sum of last k_left-1 items in D[j+1:]?
        # No, D is sorted desc. Smallest items are at the end.
        
        for j in range(start_idx, len(D)):
            if D[j] > remaining:
                continue
                
            # If we pick D[j], can we form remaining - D[j] with k_left - 1 items from D[j+1:]?
            new_remaining = remaining - D[j]
            
            # Check if max sum of k_left-1 items from D[j+1:] is >= new_remaining
            if max_sum_k(j + 1, k_left - 1) < new_remaining:
                # Even taking the largest available, we can't reach the target.
                # Since D is sorted desc, if we skip D[j] and take smaller ones, the max sum will be even smaller?
                # No, if we take smaller ones, the sum is smaller. So if the MAX sum of the best candidates is too small, 
                # then any combination of smaller candidates will also be too small.
                # Wait, D[j] is the largest candidate we are considering in this loop step?
                # We are iterating j increasing. So D[j] is getting smaller.
                # If max_sum_k(j+1, k_left-1) < new_remaining, it means even with the largest available smaller divisors, 
                # we can't reach the target. 
                # If we pick an even smaller divisor D[j+1], the new_remaining will be larger (remaining - D[j+1] > remaining - D[j]), 
                # and the pool of available divisors will be D[j+2:], which is a subset of D[j+1:]. 
                # So the max sum will be even smaller relative to the larger target.
                # Thus, we can break.
                break
            
            # Also check if it's possible to reach new_remaining with the smallest available?
            # Not strictly necessary for correctness, just pruning.
            
            current_set.append(D[j])
            res = dfs(j + 1, new_remaining, k_left - 1, current_set, deadline)
            if res:
                return res
            current_set.pop()
            
        return None

    deadline = time.time() + 50 # Safety margin
    
    for k in range(1, 20): # Try up to 19 terms, though 6 is found
        res = dfs(0, target, k, [], deadline)
        if res:
            return res
            
    # Fallback if not found (should not happen)
    return []
