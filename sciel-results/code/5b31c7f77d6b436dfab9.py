import math
import time

def construct():
    n = 256
    deadline = time.time() + 85
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Sort primes by descending max multiple value to improve pruning
    # max_val(p) = n // p * p
    primes.sort(key=lambda p: (n // p) * p, reverse=True)
    
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Precompute prime factors as bitmasks
    factor_mask = [0] * (n + 1)
    for x in range(2, n + 1):
        mask = 0
        temp = x
        for i, p in enumerate(primes):
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        factor_mask[x] = mask
        
    # 3. Precompute max value for each prime (for upper bound)
    max_val_for_prime = [0] * num_primes
    for i, p in enumerate(primes):
        max_val_for_prime[i] = (n // p) * p
        
    # Suffix sums of max values
    suffix_max = [0] * (num_primes + 1)
    for i in range(num_primes - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + max_val_for_prime[i]

    best_sum = 0
    best_set = []
    
    def dfs(idx, used_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if idx == num_primes:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        # Pruning
        if current_sum + suffix_max[idx] <= best_sum:
            return
            
        p = primes[idx]
        
        # Option 1: Skip prime p
        dfs(idx + 1, used_mask, current_sum, current_set)
        
        # Option 2: Pick a multiple of p
        # Generate candidates in descending order
        # We limit to top 5 candidates to keep branching factor low
        candidates = []
        for c in range((n // p) * p, 0, -p):
            if (factor_mask[c] & used_mask) == 0:
                candidates.append(c)
                if len(candidates) >= 5:
                    break
                    
        for c in candidates:
            new_mask = used_mask | factor_mask[c]
            new_set = current_set + [c]
            dfs(idx + 1, new_mask, current_sum + c, new_set)
            
            if time.time() > deadline:
                return

    dfs(0, 0, 0, [])
    
    # Add 1
    if 1 not in best_set:
        best_set.append(1)
        best_sum += 1
        
    return best_set
