import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    # We'll use a segmented sieve approach for phi computation
    # First, let's try to compute phi for a range and find the longest run
    
    # Strategy: Compute phi values for a range using a sieve method
    # Then scan for longest run of distinct values
    
    def compute_phi_segment(lo, hi, primes):
        """Compute phi(n) for n in [lo, hi] using segmented sieve."""
        size = hi - lo + 1
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        nums = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > hi:
                break
            # Find first multiple of p in [lo, hi]
            first = ((lo + p - 1) // p) * p
            if first < lo:
                first = lo
            
            for multiple in range(first, hi + 1, p):
                idx = multiple - lo
                phi[idx] = phi[idx] // p * (p - 1)
        
        # For remaining primes > sqrt(hi), handle via direct check
        # Actually, the standard segmented phi sieve handles this:
        # After processing all primes up to sqrt(hi), any remaining value > 1 is prime
        # and we need to apply phi adjustment
        
        # Better approach: after the loop, for each position where nums[i] > 1,
        # it's a prime factor we haven't divided out
        for i in range(size):
            rem = nums[i]
            # Wait, we need to track remaining factor
            # Let me use a different approach
        
        # Actually, let's use the standard method more carefully
        # phi[n] starts as n, and for each prime p dividing n, we multiply by (1-1/p)
        # The issue is we need to know all prime factors
        
        # Alternative: use the formula phi(n) = n * prod(1 - 1/p) for p|n
        # After dividing out all small primes, remaining > 1 is a large prime factor
        
        # Let me redo this properly
        return phi
    
    # Better segmented phi computation
    def compute_phi_segmented(lo, hi, primes):
        size = hi - lo + 1
        result = np.arange(lo, hi + 1, dtype=np.int64)  # phi values
        remaining = np.arange(lo, hi + 1, dtype=np.int64)  # remaining factor to divide
        
        for p in primes:
            if p * p > hi:
                break
            first = ((lo + p - 1) // p) * p
            for multiple in range(first, hi + 1, p):
                idx = multiple - lo
                result[idx] = result[idx] // p * (p - 1)
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
        
        # For remaining values > 1, they are prime factors
        for i in range(size):
            if remaining[i] > 1:
                p = remaining[i]
                result[i] = result[i] // p * (p - 1)
        
        return result
    
    # First, generate primes up to a reasonable limit using simple sieve
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(np.sqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]
    
    # Generate primes up to 10^6 initially
    primes_limit = 1000000
    primes = sieve_primes(primes_limit)
    
    best_k = 1
    best_s = 1
    
    # Try scanning different ranges
    # Start with a moderate range
    scan_sizes = [100000, 200000, 500000, 1000000, 2000000]
    
    for scan_size in scan_sizes:
        if time.time() > deadline - 5:
            break
        
        # Try a few starting points
        for start in [1, 1000000, 10000000, 100000000, 1000000000]:
            if time.time() > deadline - 5:
                break
            
            lo = start
            hi = min(start + scan_size - 1, 10**12)
            if lo > 10**12:
                continue
            
            try:
                phi_vals = compute_phi_segmented(lo, hi, primes)
                
                # Find longest run of distinct values
                seen = set()
                cur_k = 0
                for i in range(len(phi_vals)):
                    v = int(phi_vals[i])
                    if v not in seen:
                        seen.add(v)
                        cur_k += 1
                        if cur_k > best_k:
                            best_k = cur_k
                            best_s = lo + i - cur_k + 1
                    else:
                        seen = {v}
                        cur_k = 1
                # Reset seen after each collision for the sliding window
                # Actually, we need a proper sliding window
                # Let me redo the longest distinct run finding
                
                # Proper sliding window for longest substring of distinct values
                seen = set()
                left = 0
                for right in range(len(phi_vals)):
                    v = int(phi_vals[right])
                    while v in seen:
                        seen.discard(int(phi_vals[left]))
                        left += 1
                    seen.add(v)
                    if right - left + 1 > best_k:
                        best_k = right - left + 1
                        best_s = lo + left
                
            except Exception as e:
                continue
    
    # Also try smaller ranges near 1 where we might find good runs
    if time.time() < deadline - 10:
        for scan_size in [10000, 50000]:
            lo = 1
            hi = scan_size
            try:
                phi_vals = compute_phi_segmented(lo, hi, primes)
                seen = set()
                left = 0
                for right in range(len(phi_vals)):
                    v = int(phi_vals[right])
                    while v in seen:
                        seen.discard(int(phi_vals[left]))
                        left += 1
                    seen.add(v)
                    if right - left + 1 > best_k:
                        best_k = right - left + 1
                        best_s = lo + left
            except Exception:
                continue
    
    # Ensure constraints are met
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, int(10**12 - best_s) + 1)
    if best_s < 1:
        best_s = 1
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
    
    return {'start': best_s, 'length': best_k}
