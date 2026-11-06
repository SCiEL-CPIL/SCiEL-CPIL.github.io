import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)
    
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    valid_divs = sorted([d for d in divisors if d > 12])
    
    exp_vecs = []
    for d in valid_divs:
        vec = [0] * 5
        for j, p in enumerate(primes):
            while d % p == 0:
                vec[j] += 1
                d //= p
        exp_vecs.append(vec)
        
    n_valid = len(valid_divs)
    
    # Precompute log of valid_divs for pruning?
    # Or just use the product of largest k.
    
    def dfs(start_idx, remaining_exp, count, current_factors):
        if time.time() - start_time > timeout:
            return None
        
        if all(x == 0 for x in remaining_exp):
            if count > 0:
                return list(current_factors)
            return None
        
        if count >= 12:
            return None
            
        # Additional Pruning:
        # If the product of the largest (12 - count) candidates in [0..start_idx] is < remaining_value, fail.
        # We don't track remaining_value, only remaining_exp.
        # We can reconstruct remaining_value from exp? 
        # val = 2^e0 * 3^e1 ...
        # This is fast.
        
        # Let's calculate remaining_value
        rem_val = 1
        for j in range(5):
            if remaining_exp[j] > 0:
                rem_val *= primes[j] ** remaining_exp[j]
                
        slots_left = 12 - count
        # Max product we can form with slots_left factors from 0..start_idx
        # Take the largest slots_left candidates
        # Since valid_divs is sorted ascending, they are at the end of 0..start_idx
        # Indices: start_idx, start_idx-1, ...
        
        max_prod = 1
        # We need at most slots_left factors.
        # If slots_left >= start_idx + 1, we can take all.
        # Else take the largest slots_left.
        
        # Optimization: 
        # If rem_val > 1, we must pick at least one more.
        # If the product of the largest slots_left candidates is < rem_val, we fail.
        
        # Calculate product of largest min(slots_left, start_idx+1) candidates
        k = min(slots_left, start_idx + 1)
        # Candidates are valid_divs[start_idx - k + 1 : start_idx + 1]
        # Product of these
        # This calculation inside DFS might be slow if done every step.
        # But k is small (<=12).
        
        # Let's just do a quick check: 
        # If rem_val is large, and candidates are small, we fail.
        # Specifically, if valid_divs[start_idx] ** slots_left < rem_val, we fail.
        # This is a lower bound on the max product? No, it's an upper bound if we assume all are max.
        # If (max_candidate)^slots_left < rem_val, then even if we picked the max candidate slots_left times, we couldn't reach rem_val.
        # Since candidates are distinct, the product is actually smaller.
        # So this is a valid pruning condition.
        
        if valid_divs[start_idx] ** slots_left < rem_val:
            return None
            
        for i in range(start_idx, -1, -1):
            if time.time() - start_time > timeout:
                return None
                
            c_vec = exp_vecs[i]
            
            if all(remaining_exp[j] >= c_vec[j] for j in range(5)):
                new_remaining = [remaining_exp[j] - c_vec[j] for j in range(5)]
                current_factors.append(valid_divs[i])
                
                res = dfs(i - 1, new_remaining, count + 1, current_factors)
                
                current_factors.pop()
                
                if res:
                    return res
                    
        return None

    for M_idx in range(n_valid):
        M = valid_divs[M_idx]
        
        # Pruning in main loop:
        # Max product of 12 largest candidates in pool [0..M_idx]
        k = min(12, M_idx + 1)
        # Product of valid_divs[M_idx - k + 1 : M_idx + 1]
        max_possible_prod = 1
        for j in range(M_idx - k + 1, M_idx + 1):
            max_possible_prod *= valid_divs[j]
            
        if max_possible_prod < target:
            continue
            
        sol = dfs(M_idx, target_exp, 0, [])
        
        if sol:
            return sol
            
        if time.time() - start_time > timeout:
            break
            
    return [target]
