# EVOLVE-BLOCK-START
import math
import time
import itertools

def construct():
    n = 256
    
    # 1. Identify primes
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # 2. Split primes into small and large
    # Threshold: sqrt(256) = 16. Primes > 16 are "large".
    small_primes = [p for p in primes if p <= 16]
    large_primes = [p for p in primes if p > 16]
    
    # Map large primes to bit indices
    large_prime_index = {p: i for i, p in enumerate(large_primes)}
    
    # 3. Generate candidates for each small prime
    # Candidate: (value, blocked_mask)
    groups = []
    for p in small_primes:
        candidates = []
        # Option: Pick nothing (0)
        candidates.append((0, 0))
        
        # Find all multiples m of p <= n such that SPF(m) == p
        for m in range(p, n + 1, p):
            # Check if m has any prime factor < p
            has_small_factor = False
            for q in small_primes:
                if q >= p:
                    break
                if m % q == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                # m is a valid candidate for SPF p
                mask = 0
                for lp in large_primes:
                    if m % lp == 0:
                        mask |= (1 << large_prime_index[lp])
                candidates.append((m, mask))
        
        groups.append(candidates)
    
    # 4. Iterate through all combinations
    best_sum = 0
    best_combo = None
    
    for combo in itertools.product(*groups):
        # Check validity among small-SPF numbers
        valid = True
        chosen_vals = []
        for val, mask in combo:
            if val != 0:
                chosen_vals.append(val)
        
        # Check pairwise gcd
        for i in range(len(chosen_vals)):
            for j in range(i + 1, len(chosen_vals)):
                if math.gcd(chosen_vals[i], chosen_vals[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Calculate sum
        current_sum = 1
        blocked_mask = 0
        for val, mask in combo:
            current_sum += val
            blocked_mask |= mask
        
        # Add unblocked large primes
        for i, lp in enumerate(large_primes):
            if not (blocked_mask & (1 << i)):
                current_sum += lp
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_combo = combo
            
    # Reconstruct result
    result = [1]
    if best_combo:
        for val, mask in best_combo:
            if val != 0:
                result.append(val)
        
        # Re-calculate blocked mask for result construction
        blocked_mask = 0
        for val, mask in best_combo:
            blocked_mask |= mask
            
        for i, lp in enumerate(large_primes):
            if not (blocked_mask & (1 << i)):
                result.append(lp)
                
    return result
# EVOLVE-BLOCK-END
