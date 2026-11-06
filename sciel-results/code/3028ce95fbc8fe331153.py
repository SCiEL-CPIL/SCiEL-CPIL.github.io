"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def sieve_primes(limit):
        """Return a boolean array where arr[i] is True if i is prime."""
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return is_prime
    
    def compute_phi_sieve_numpy(limit):
        """Compute phi for all numbers up to limit using precomputed primes."""
        phi = np.arange(limit + 1, dtype=np.int64)
        is_prime = sieve_primes(limit)
        primes = np.where(is_prime)[0]
        for p in primes:
            if p > limit:
                break
            indices = np.arange(p, limit + 1, p)
            phi[indices] -= phi[indices] // p
        return phi
    
    def find_longest_distinct_run(phi_vals):
        """
        Find the longest contiguous subsequence with distinct values.
        Returns (max_length, start_index).
        """
        n = len(phi_vals)
        if n < 2:
            return 1, 1 if n > 1 else 0
            
        last_seen = {}
        start = 1
        max_len = 0
        max_start = 1
        
        for i in range(1, n):
            val = phi_vals[i]
            if val in last_seen and last_seen[val] >= start:
                start = last_seen[val] + 1
            last_seen[val] = i
            
            current_len = i - start + 1
            if current_len > max_len:
                max_len = current_len
                max_start = start
                
        return max_len, max_start

    def compute_phi_segmented(start, length, primes):
        """
        Compute phi for numbers in [start, start+length-1] using segmented sieve.
        primes is a list/array of all primes up to sqrt(start+length).
        """
        n = length
        phi = np.arange(start, start + n, dtype=np.int64)
        
        for p in primes:
            if p * p > start + n - 1:
                break
            # Find first multiple of p in [start, start+n-1]
            first = ((start + p - 1) // p) * p
            indices = np.arange(first, start + n, p) - start
            phi[indices] -= phi[indices] // p
        
        return phi

    best_k = 1
    best_s = 1
    
    # Phase 1: Full sieve up to a moderate limit
    full_limit = 10000000
    try:
        if time_remaining() > 20.0:
            phi_vals = compute_phi_sieve_numpy(full_limit)
            k, s = find_longest_distinct_run(phi_vals)
            if k > 200:
                k = 200
            if k > best_k:
                best_k = k
                best_s = s
    except Exception:
        pass
    
    # Phase 2: Segmented sieve for larger ranges
    # Use primes up to sqrt(10^12) ~ 10^6
    sqrt_max = int(math.isqrt(10**12)) + 1
    if time_remaining() > 30.0:
        try:
            is_prime_small = sieve_primes(sqrt_max)
            primes = np.where(is_prime_small)[0]
            
            # Try several segments at different locations
            segment_size = 10000000  # 10 million per segment
            
            # Candidate start positions: spread across [1, 10^12]
            # Focus on regions where distinct totients are more likely
            # (e.g., near products of small primes, or random locations)
            candidate_starts = []
            
            # Some structured candidates
            # Near 1
            candidate_starts.append(1)
            # Near powers of 2
            for e in range(1, 40):
                candidate_starts.append(2**e - 10**6)
            # Near factorials
            factorial = 1
            for i in range(1, 20):
                factorial *= i
                if factorial > 10**12:
                    break
                candidate_starts.append(factorial - 10**6)
            # Some random-ish structured locations
            for i in range(1, 20):
                candidate_starts.append(i * 10**11 - 10**6)
            for i in range(1, 10):
                candidate_starts.append(i * 10**10 - 10**6)
            
            # Filter valid starts
            valid_starts = []
            for cs in candidate_starts:
                if cs < 1:
                    cs = 1
                if cs + segment_size - 1 > 10**12:
                    cs = max(1, 10**12 - segment_size + 1)
                valid_starts.append(cs)
            
            # Also add some pseudo-random starts
            rng = np.random.default_rng(42)
            for _ in range(50):
                cs = int(rng.integers(1, 10**12 - segment_size + 1))
                valid_starts.append(cs)
            
            # Deduplicate and sort
            valid_starts = sorted(set(valid_starts))
            
            for cs in valid_starts:
                if time_remaining() < 15.0:
                    break
                if cs + segment_size - 1 > 10**12:
                    cs = max(1, 10**12 - segment_size + 1)
                
                try:
                    phi_seg = compute_phi_segmented(cs, segment_size, primes)
                    k, s_offset = find_longest_distinct_run(phi_seg)
                    
                    if k > 200:
                        k = 200
                        s_actual = cs + s_offset - 1
                    else:
                        s_actual = cs + s_offset - 1
                    
                    if s_actual < 1:
                        s_actual = 1
                    if s_actual + k - 1 > 10**12:
                        k = min(k, 10**12 - s_actual + 1)
                    
                    if k > best_k:
                        best_k = k
                        best_s = s_actual
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # Ensure valid return
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    # Final safety cap
    if best_k > 200:
        best_k = 200
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_s = max(1, 10**12 - best_k + 1)
        
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
