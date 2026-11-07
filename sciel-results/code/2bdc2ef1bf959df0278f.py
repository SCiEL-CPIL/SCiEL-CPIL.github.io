import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin: 80s out of 90s limit
    
    n = 256
    
    # Precompute factor masks for each number 1..256
    # Each prime factor gets a bit position
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Assign bit positions to primes
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # Compute factor mask for each number
    factor_mask = [0] * (n + 1)
    for i in range(2, n + 1):
        mask = 0
        x = i
        for p in primes:
            if p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_bit[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_bit[x])
        factor_mask[i] = mask
    
    # factor_mask[1] = 0 (no prime factors, coprime to everything)
    
    def is_compatible(val, current_set):
        """Check if val is coprime to all elements in current_set."""
        fm = factor_mask[val]
        if fm == 0:  # val is 1, always compatible
            return True
        for v in current_set:
            if (fm & factor_mask[v]) != 0:
                return False
        return True
    
    # Initial solution: 1 plus largest prime power for each prime
    result_set = {1}
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        # Check if power is compatible with current set
        # Since we're adding prime powers of distinct primes, they should be coprime
        # But let's verify
        if is_compatible(power, result_set):
            result_set.add(power)
    
    best_set = set(result_set)
    best_sum = sum(best_set)
    
    # Local search: try to improve by swaps and additions
    # Use a priority-based approach: try to add high-value candidates first
    candidates = list(range(n, 0, -1))  # Descending order
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Try adding each candidate not in current set
        for val in candidates:
            if time.time() >= deadline - 2.0:
                break
            if val in best_set:
                continue
            if is_compatible(val, best_set):
                new_sum = best_sum + val
                if new_sum > best_sum:
                    best_set.add(val)
                    best_sum = new_sum
                    improved = True
        
        # Phase 2: Try swaps - remove one element, add one or more
        for remove_val in list(best_set):
            if time.time() >= deadline - 2.0:
                break
            
            reduced_set = best_set - {remove_val}
            reduced_sum = best_sum - remove_val
            
            # Try adding each candidate not in reduced_set
            for add_val in candidates:
                if time.time() >= deadline - 2.0:
                    break
                if add_val in reduced_set:
                    continue
                if is_compatible(add_val, reduced_set):
                    new_sum = reduced_sum + add_val
                    if new_sum > best_sum:
                        best_set = reduced_set
                        best_set.add(add_val)
                        best_sum = new_sum
                        improved = True
                        break
    
    # Phase 3: Try 2-for-1 swaps (remove one, add two)
    if time.time() < deadline - 2.0:
        for remove_val in list(best_set):
            if time.time() >= deadline - 2.0:
                break
            
            reduced_set = best_set - {remove_val}
            reduced_sum = best_sum - remove_val
            
            # Try adding two candidates
            for add1 in candidates:
                if time.time() >= deadline - 2.0:
                    break
                if add1 in reduced_set:
                    continue
                if not is_compatible(add1, reduced_set):
                    continue
                
                temp_set = reduced_set | {add1}
                temp_sum = reduced_sum + add1
                
                for add2 in candidates:
                    if time.time() >= deadline - 2.0:
                        break
                    if add2 in temp_set:
                        continue
                    if not is_compatible(add2, temp_set):
                        continue
                    
                    new_sum = temp_sum + add2
                    if new_sum > best_sum:
                        best_set = temp_set
                        best_set.add(add2)
                        best_sum = new_sum
                        improved = True
                        break
    
    # Convert to list of Python ints
    return [int(x) for x in best_set]
