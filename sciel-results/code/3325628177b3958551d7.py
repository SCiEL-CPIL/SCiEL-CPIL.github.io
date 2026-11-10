import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, length):
        """
        Compute phi(x) for x in [start, start+length-1] using segmented sieve.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi values to the numbers themselves
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        # rem tracks the remaining part of the number to detect large prime factors
        rem = np.arange(start, start + length, dtype=np.int64)
        
        # Process small primes
        for p in primes:
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            
            # Iterate over multiples
            # Use numpy slicing for speed?
            # Indices of multiples in the segment:
            # first, first+p, ..., < start+length
            # idx = first - start, idx + p, ...
            if first >= start + length:
                continue
                
            # Create indices
            # Note: p can be large, so the number of iterations is small for large p
            # But for small p, it's many iterations.
            # Python loop is slow. Let's try to vectorize where possible, 
            # but the "while rem % p == 0" part is tricky to vectorize cleanly without masks.
            # Given length is ~1000-5000, and number of primes is 78k, 
            # most primes will have 0 or 1 multiple.
            # For small primes, many multiples.
            
            # Optimization: Only process primes where p <= start+length-1? 
            # No, p must be a factor.
            
            # Let's stick to a loop for correctness and simplicity, 
            # but only if the prime actually divides any number in the range.
            
            # For very large p, first == start + (p - start % p) % p.
            # If p > length, there is at most 1 multiple.
            
            # To speed up, we can skip primes p > start+length-1? 
            # No, a prime p > length can still divide a number in the range.
            # But if p > 10^6, we handle it in the "large prime" step.
            # So we only loop over primes in `primes` (up to 10^6).
            
            # For each multiple:
            # We need to update phi and rem.
            # Vectorized approach for the update:
            # indices = np.arange(first, start+length, p) - start
            # mask = rem[indices] % p == 0
            # This is safe.
            
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            # Check which of these are actually divisible by p in rem
            # Actually, by definition, they are multiples of p.
            # So rem[indices] % p == 0 is always true initially.
            # But we divide rem by p.
            # We need to apply the phi factor once per distinct prime factor.
            # So we just do:
            # phi_vals[indices] = phi_vals[indices] // p * (p - 1)
            # And divide rem by p as many times as possible.
            
            # The division of rem is the hard part to vectorize (repeated division).
            # However, for most numbers, the exponent of p is 1.
            # We can do:
            # rem[indices] //= p
            # Then check again? 
            # rem2 = rem[indices] % p == 0
            # rem[indices[rem2]] //= p
            # ...
            # This is complex.
            
            # Given the time limit, and that length is small (e.g., 1000),
            # a Python loop over the indices might be acceptable if we only do it for primes that have multiples.
            # Number of multiples of p in range L is L/p.
            # Sum of L/p for p in primes is L * log(log(limit)).
            # For L=1000, this is small.
            
            for idx in indices:
                # Update phi
                phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                # Update rem
                while rem[idx] % p == 0:
                    rem[idx] //= p
        
        # Handle large prime factors
        # If rem[i] > 1, it is a prime factor > 10^6
        # Since n <= 10^12, rem[i] can be at most 10^12.
        # But n = (small factors) * rem[i].
        # rem[i] is prime.
        mask = rem > 1
        if np.any(mask):
            # For these indices, apply the factor
            # phi_vals[idx] = phi_vals[idx] // rem[idx] * (rem[idx] - 1)
            # Vectorized:
            phi_vals[mask] = phi_vals[mask] // rem[mask] * (rem[mask] - 1)
            
        return phi_vals

    def find_best_run_sliding(start, length, max_k=200):
        """
        Find the longest run of distinct phi values in [start, start+length)
        """
        if length <= 0:
            return 0, start
        
        phis = compute_phi_segment(start, length)
        
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(length):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = start + left
            if best_k >= max_k:
                break
        
        return min(best_k, max_k), best_s

    best_k = 1
    best_s = 1
    
    # Baseline check
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 1000, 200)
        if k > best_k:
            best_k, best_s = k, s

    random.seed(42)
    
    # Search in random ranges
    # Longer segments allow finding longer runs, but cost more time.
    # Let's try segment length 2000.
    seg_len = 2000
    
    while time_left() > 5:
        start = random.randint(10**6, 10**12 - seg_len)
        k, s = find_best_run_sliding(start, seg_len, 200)
        if k > best_k:
            best_k, best_s = k, s
            # If we found a long run, try to extend it
            if best_k > 10:
                # Extend left and right
                # Left extension
                ext_start = max(1, best_s - 1000)
                ext_len = best_s - ext_start + best_k + 1000
                k_ext, s_ext = find_best_run_sliding(ext_start, ext_len, 200)
                if k_ext > best_k:
                    best_k, best_s = k_ext, s_ext
                
                # Right extension
                ext_start_r = best_s
                ext_end_r = min(10**12, best_s + best_k + 1000)
                ext_len_r = ext_end_r - ext_start_r + 1
                k_ext_r, s_ext_r = find_best_run_sliding(ext_start_r, ext_len_r, 200)
                if k_ext_r > best_k:
                    best_k, best_s = k_ext_r, s_ext_r

    return {'start': best_s, 'length': best_k}
