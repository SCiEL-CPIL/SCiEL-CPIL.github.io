"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s limit
    
    # Precompute primes up to 10^6 for efficient factorization
    # For n <= 10^12, after removing all prime factors <= 10^6,
    # the remainder is either 1 or prime (since p*q > 10^12 if p,q > 10^6)
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    if limit_p >= 1:
        is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = 0
    
    # Extract primes list for trial division
    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]
    
    def phi(x):
        """Compute Euler's totient function using trial division by primes."""
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def segmented_phi(s, k):
        """
        Compute phi(s), phi(s+1), ..., phi(s+k-1) using segmented sieve.
        Returns list of k totient values.
        """
        if k <= 0:
            return []
        
        # Initialize arrays for the segment [s, s+k-1]
        # phi_vals[i] will hold phi(s+i)
        # remaining[i] will hold the remaining part after removing small prime factors
        phi_vals = [0] * k
        remaining = [0] * k
        
        for i in range(k):
            phi_vals[i] = s + i
            remaining[i] = s + i
        
        # For each prime p <= sqrt(s+k-1), mark multiples in the segment
        max_n = s + k - 1
        sqrt_max = int(math.isqrt(max_n)) + 1
        
        for p in primes:
            if p > sqrt_max:
                break
            
            # Find the first multiple of p in [s, s+k-1]
            first_mult = ((s + p - 1) // p) * p
            if first_mult < p * p:
                # Skip if the first multiple is p itself (we only want composites)
                # Actually, we need to handle all multiples, not just composites
                # The totient formula works for all numbers
                pass
            
            for m in range(first_mult, max_n + 1, p):
                idx = m - s
                if remaining[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors > sqrt_max
        for i in range(k):
            if remaining[i] > 1:
                # remaining[i] is a prime factor > sqrt_max
                phi_vals[i] -= phi_vals[i] // remaining[i]
        
        return phi_vals
    
    def check_run_segmented(s, k):
        """Check if phi(s), ..., phi(s+k-1) are pairwise distinct using segmented sieve."""
        phi_values = segmented_phi(s, k)
        seen = set()
        for val in phi_values:
            if val in seen:
                return False
            seen.add(val)
        return True
    
    def find_max_run_from_start(s):
        """Find the maximum run length starting at s using segmented sieve."""
        # Use a sliding window approach with segmented sieve
        # Start with a small window and expand
        max_k = 1
        seen = set()
        
        # Compute phi for a larger window at once for efficiency
        window_size = 250  # Slightly larger than 200 to have margin
        
        # We'll compute in chunks
        offset = 0
        while s + offset + window_size <= 10**12 + window_size:
            # Compute phi for the next chunk
            chunk_end = min(s + offset + window_size, s + 10**12)
            chunk_size = chunk_end - s - offset
            
            if chunk_size <= 0:
                break
            
            phi_values = segmented_phi(s + offset, chunk_size)
            
            for i, val in enumerate(phi_values):
                if val in seen:
                    return max_k
                seen.add(val)
                max_k += 1
            
            offset += chunk_size
            
            if time.time() > DEADLINE - 5:
                break
            if max_k >= 200:
                break
        
        return max_k
    
    # Phase 1: Try small numbers (fast, likely short runs)
    best_k = 1
    best_s = 1
    
    # Check small range first using segmented sieve
    small_limit = 10000
    # Compute all phi values in the small range at once
    small_phi = segmented_phi(1, small_limit)
    
    for s in range(1, small_limit - 200):
        seen, k = set(), 0
        while s + k - 1 <= small_limit and small_phi[s + k - 1] not in seen:
            seen.add(small_phi[s + k - 1])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if best_k >= 200:
            break
    
    if time.time() > DEADLINE - 10:
        return {'start': best_s, 'length': best_k}
    
    # Phase 2: Random sampling of large numbers using segmented sieve
    # For large n, totient collisions are rare, so long runs are more likely
    random.seed(42)
    
    # Sample from different ranges
    num_attempts = 0
    while time.time() < DEADLINE - 10:
        num_attempts += 1
        
        # Choose a random starting point in a large range
        # Focus on ranges where n is large (10^8 to 10^12)
        if num_attempts % 3 == 0:
            s = random.randint(10**8, 10**12 - 200)
        elif num_attempts % 3 == 1:
            s = random.randint(10**6, 10**8)
        else:
            s = random.randint(10**4, 10**6)
        
        # Use segmented sieve to check for long runs
        # First, compute phi for a window of 200
        phi_values = segmented_phi(s, 200)
        
        # Check for duplicates
        seen = set()
        k = 0
        for val in phi_values:
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break
    
    # Phase 3: Local optimization around best start found using segmented sieve
    if best_k < 200 and time.time() < DEADLINE - 15:
        # Try neighbors of the best start
        for offset in range(-50, 51):
            if time.time() > DEADLINE - 5:
                break
            s = best_s + offset
            if s < 1 or s + 200 > 10**12:
                continue
            
            phi_values = segmented_phi(s, 200)
            
            seen = set()
            k = 0
            for val in phi_values:
                if val in seen:
                    break
                seen.add(val)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    break
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
