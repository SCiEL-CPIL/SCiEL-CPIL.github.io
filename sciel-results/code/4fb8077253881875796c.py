import time
from math import gcd
from fractions import Fraction

def _precompute():
    N = 512
    primes = []
    for i in range(2, N):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * N
    for a in range(1, N):
        m = 0
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << prime_index[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            m |= (1 << prime_index[temp])
        masks[a] = m
        
    values = [Fraction(0)] * N
    for a in range(1, N):
        values[a] = Fraction(1, 512 - a)
        
    # Map mask to the best a (largest a) for that mask
    mask_to_best_a = {}
    for a in range(511, 0, -1):
        m = masks[a]
        if m not in mask_to_best_a:
            mask_to_best_a[m] = a
            
    # List of (mask, a, value)
    candidates = []
    for m, a in mask_to_best_a.items():
        candidates.append((m, a, values[a]))
        
    # Sort by value descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    return candidates, primes

def _branch_and_bound(candidates, deadline):
    n = len(candidates)
    # Precompute suffix sums of values for pruning
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + candidates[i][2]
        
    best_sum = Fraction(0)
    best_set = set()
    
    # Recursive search
    # To avoid recursion limit issues or overhead, we can use a stack or optimized recursion
    # Given n is small, recursion is fine.
    
    def dfs(idx, used_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
            
        # Pruning
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Option 1: Skip candidate idx
        dfs(idx + 1, used_mask, current_sum, current_set)
        
        # Option 2: Take candidate idx if compatible
        m, a, val = candidates[idx]
        if (m & used_mask) == 0:
            current_set.append(a)
            dfs(idx + 1, used_mask | m, current_sum + val, current_set)
            current_set.pop()
            
    dfs(0, 0, Fraction(0), [])
    
    return list(best_set), best_sum

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    candidates, primes = _precompute()
    
    best_S, best_score = _branch_and_bound(candidates, deadline)
    
    return best_S
