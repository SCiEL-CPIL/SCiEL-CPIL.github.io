"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a).
    
    Uses Branch-and-Bound to solve the Maximum Weight Independent Set problem
    on the coprimality conflict graph.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    N = 512
    MAX_A = 511
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # For each candidate a (1..511), compute its prime factor bitmask
    def prime_factor_mask(n, primes, prime_to_idx):
        mask = 0
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << prime_to_idx[p])
                while n % p == 0:
                    n //= p
        if n > 1:
            mask |= (1 << prime_to_idx[n])
        return mask
    
    masks = [0] * (MAX_A + 1)
    weights = [0.0] * (MAX_A + 1)
    
    for a in range(1, MAX_A + 1):
        masks[a] = prime_factor_mask(a, primes, prime_to_idx)
        weights[a] = 1.0 / (N - a)
    
    # Build conflict graph using bitmasks
    # conflicts[i] is a bitmask of j such that gcd(i, j) > 1
    conflicts = [0] * (MAX_A + 1)
    for i in range(1, MAX_A + 1):
        for j in range(i + 1, MAX_A + 1):
            if (masks[i] & masks[j]) != 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
                
    # Sort candidates by weight descending
    candidates = list(range(1, MAX_A + 1))
    candidates.sort(key=lambda x: weights[x], reverse=True)
    
    # Precompute suffix sums of weights for the sorted candidates
    suffix_sum = [0.0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + weights[candidates[i]]
        
    best_weight = 0.0
    best_set = []
    
    def search(k, current_weight, current_mask):
        nonlocal best_weight, best_set, deadline, start_time
        
        # Time check
        if time.time() > deadline:
            return
            
        if k == len(candidates):
            if current_weight > best_weight:
                best_weight = current_weight
                # Reconstruct set from current_mask
                best_set = []
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if idx <= MAX_A:
                        best_set.append(idx)
                    m ^= lsb
            return
        
        v = candidates[k]
        
        # If v is not in current_mask, it was removed by a previous choice
        if not (current_mask & (1 << v)):
            search(k + 1, current_weight, current_mask)
            return
            
        # Option 1: Include v
        # New mask: remove v and all its conflicts from current_mask
        new_mask_include = current_mask & ~(1 << v) & ~conflicts[v]
        
        # Upper bound: current_weight + weights[v] + sum of remaining weights (suffix_sum[k+1])
        if current_weight + weights[v] + suffix_sum[k+1] > best_weight:
            search(k + 1, current_weight + weights[v], new_mask_include)
            
        # Option 2: Exclude v
        # New mask: remove v from current_mask
        new_mask_exclude = current_mask & ~(1 << v)
        
        # Upper bound: current_weight + sum of remaining weights (suffix_sum[k+1])
        if current_weight + suffix_sum[k+1] > best_weight:
            search(k + 1, current_weight, new_mask_exclude)
            
    # Initial mask: all candidates 1..511 are available
    initial_mask = 0
    for a in range(1, MAX_A + 1):
        initial_mask |= (1 << a)
        
    # Start search
    search(0, 0.0, initial_mask)
    
    return best_set
# EVOLVE-BLOCK-END
