"""Improved program using segmented sieve for p-smooth checking."""
import math
import time

# EVOLVE-BLOCK-START

def sieve_primes(limit):
    """Sieve of Eratosthenes for primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.2e18."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def check_p_smooth_segmented(p, primes, max_k=200):
    """
    Check the length of the run of p-smooth numbers starting at p^2 + 1.
    Uses a segmented sieve approach: divide out all prime factors <= p from the interval [p^2+1, p^2+max_k].
    If any number remains > 1, it has a prime factor > p.
    """
    start = p * p + 1
    # We only need to process primes q <= p
    # Find index of last prime <= p
    # Since primes is sorted, we can binary search or just iterate if we are careful
    # For efficiency in the main loop, we'll pass a slice or index
    
    # Initialize remainders
    rems = [0] * max_k
    for i in range(max_k):
        rems[i] = start + i
        
    # Sieve out factors
    # We need primes q such that q <= p
    # Also, we only need to consider q up to sqrt(start + max_k) for the "marking" part?
    # No, we need to divide by ALL primes q <= p.
    # However, if q > sqrt(start + max_k), q can appear at most once in each number.
    # Actually, if q > sqrt(N), N can have at most one factor q.
    # But we are checking if the REMAINDER is 1.
    # So we must divide by all q <= p.
    
    # Optimization: 
    # If q > sqrt(start + max_k), then q divides at most one number in the interval (since interval length 200 < q).
    # Actually, if q > 200, it divides at most one number.
    # If q <= 200, it can divide multiple.
    
    # Let's just iterate through primes <= p.
    # To avoid iterating through all primes up to 10^7 for every p, 
    # we can precompute or use the fact that we are iterating p in increasing order.
    
    # For this function, let's assume we have the list of primes up to p.
    # We will pass the list and the cutoff index.
    
    # In the main loop, we will maintain the index.
    
    # For now, let's implement the core logic.
    # We need to know which primes to use.
    # The caller should provide `primes` and `prime_idx` (number of primes to use).
    
    pass # Placeholder, logic moved to construct for efficiency

def construct():
    start_time = time.time()
    time_limit = 115
    
    # Precompute primes up to a reasonable limit
    # 10^7 takes ~1s and 10MB. 10^6 takes ~0.1s and 1MB.
    # Let's try 10^6 first, then extend if fast.
    limit = 10**6
    all_primes = sieve_primes(limit)
    
    best_k = 0
    best_p = 3
    
    # We will iterate through primes
    # For each prime p, we need to check p^2+1 ... p^2+k
    
    # To optimize, we can precompute the "small" primes that will be used for sieving.
    # For a given p, we use all primes q <= p.
    
    # Efficient check function
    def get_run_length(p, primes, prime_count):
        """
        Returns the length of the p-smooth run starting at p^2+1.
        primes: list of primes
        prime_count: number of primes in 'primes' to use (i.e., primes[:prime_count] are <= p)
        """
        start = p * p + 1
        max_k = 200
        rems = [start + i for i in range(max_k)]
        
        # Sieve with primes
        # Only primes q <= p
        # Note: if q > start, q cannot divide start+i for i < 200 (unless start+i == q, but q <= p < start)
        # So we only need primes q <= min(p, start) which is just p.
        
        for idx in range(prime_count):
            q = primes[idx]
            if q > p:
                break
            
            # Find first index i in [0, max_k) such that (start + i) % q == 0
            # start % q = r
            # We need (r + i) % q == 0 => i % q == (-r) % q
            r = start % q
            # smallest i >= 0 such that i % q == (-r) % q
            i0 = (-r) % q
            
            # Iterate i = i0, i0+q, i0+2q, ... < max_k
            for i in range(i0, max_k, q):
                # Divide rems[i] by q as many times as possible
                while rems[i] % q == 0:
                    rems[i] //= q
                    
        # Check if all rems are 1
        k = 0
        for i in range(max_k):
            if rems[i] == 1:
                k += 1
            else:
                break
        return k

    # Phase 1: Search primes up to 10^6
    # We iterate through primes.
    # For each prime p, prime_count is the index of p in all_primes + 1.
    
    for idx, p in enumerate(all_primes):
        if time.time() - start_time > time_limit:
            break
            
        # Optimization: if p is small, the run is likely short.
        # But we want to maximize k.
        
        # We can skip some primes if we are confident?
        # No, let's just check them. The segmented sieve is fast.
        
        # prime_count is idx + 1
        k = get_run_length(p, all_primes, idx + 1)
        
        if k > best_k:
            best_k = k
            best_p = p
            
        # Early termination heuristic?
        # If we find a very large k, we might want to check neighbors.
        # But for now, just scan.

    # If time remains, extend sieve to 10^7
    if time.time() - start_time < time_limit - 2:
        limit2 = 10**7
        # We already have primes up to 10^6.
        # We can just sieve up to 10^7.
        # Re-sieving is cheap.
        all_primes_large = sieve_primes(limit2)
        
        # Continue search from 10^6 to 10^7
        # Find start index
        start_idx = len(all_primes)
        
        for idx in range(start_idx, len(all_primes_large)):
            if time.time() - start_time > time_limit:
                break
            p = all_primes_large[idx]
            
            # Use all_primes_large for the sieve
            k = get_run_length(p, all_primes_large, idx + 1)
            
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
