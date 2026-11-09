import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    # 1. Sieve primes up to 10^6 (sqrt(10^12))
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    prime_set = set(primes)

    def is_probable_prime(n):
        """Miller-Rabin primality test for 64-bit integers."""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Test with deterministic witnesses for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
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

    def compute_phi_segment(start, end):
        """
        Computes phi(n) for n in [start, end] using segmented sieve.
        Returns a numpy array of phi values (int64).
        """
        size = end - start + 1
        vals = np.arange(start, end + 1, dtype=np.int64)
        phis = vals.copy()
        rems = vals.copy()
        
        # Process small prime factors
        for p in primes:
            if p * p > end:
                break
            
            first_mult = ((start + p - 1) // p) * p
            indices = np.arange(first_mult, end + 1, p, dtype=np.int64)
            if len(indices) == 0:
                continue
            idx_array = indices - start
            
            mask = rems[idx_array] % p == 0
            valid_idx = idx_array[mask]
            
            if len(valid_idx) == 0:
                continue
            
            phis[valid_idx] -= phis[valid_idx] // p
            
            while True:
                still_div = rems[valid_idx] % p == 0
                if not np.any(still_div):
                    break
                valid_idx = valid_idx[still_div]
                rems[valid_idx] //= p
                    
        # Process remaining large prime factors
        large_prime_mask = rems > 1
        if np.any(large_prime_mask):
            large_indices = np.where(large_prime_mask)[0]
            p_vals = rems[large_indices]
            phis[large_indices] -= phis[large_indices] // p_vals
            
        return phis

    def find_longest_run(phi_vals):
        """
        Finds the longest run of distinct values in phi_vals.
        Returns (max_length, relative_start_index).
        """
        best_k = 1
        best_s_rel = 0
        n = len(phi_vals)
        
        if n == 0:
            return 0, 0

        last_seen = {}
        current_start_rel = 0
        
        for i in range(n):
            val = phi_vals[i]
            
            if val in last_seen:
                last_idx = last_seen[val]
                if last_idx >= current_start_rel:
                    current_start_rel = last_idx + 1
            
            last_seen[val] = i
            current_len = i - current_start_rel + 1
            
            if current_len > best_k:
                best_k = current_len
                best_s_rel = current_start_rel
                
            if best_k >= 200:
                return best_k, best_s_rel
                
        return best_k, best_s_rel

    def generate_candidate_starts(num_candidates, seed):
        """
        Generate candidate starting points by finding numbers with large prime factors.
        Strategy: Pick random primes p in [10^6, 10^12/2], then consider numbers
        of the form p * q where q is a small prime or 1. These numbers have
        phi(p*q) = (p-1)*(q-1) which is "large" and less likely to collide
        with nearby totients.
        """
        rng = random.Random(seed)
        candidates = []
        
        # Use primes from our sieve and generate large primes
        small_primes = [p for p in primes if p > 100]
        
        for _ in range(num_candidates):
            if time.time() - start_time > deadline - 5:
                break
            
            # Strategy 1: p * q where p is large prime, q is small prime
            if rng.random() < 0.5:
                # Pick a large prime p in [10^6, 10^12/3]
                # Use random 30-40 bit primes
                bits = rng.randint(28, 38)
                p = 1 << (bits - 1) | rng.randint(0, (1 << (bits - 1)) - 1) | 1
                # Make it odd and check primality
                for _ in range(20):
                    if is_probable_prime(p):
                        break
                    p += 2
                else:
                    p = None
                
                if p is not None and p > 10**6:
                    # Pick a small prime q
                    q = rng.choice(small_primes)
                    if q < p:
                        n = p * q
                        if n <= 10**12 and n >= 10**8:
                            candidates.append(n - rng.randint(0, 50))
                    else:
                        n = p * q
                        if n <= 10**12 and n >= 10**8:
                            candidates.append(n - rng.randint(0, 50))
            
            # Strategy 2: Prime itself (phi(p) = p-1)
            elif rng.random() < 0.3:
                bits = rng.randint(30, 40)
                p = 1 << (bits - 1) | rng.randint(0, (1 << (bits - 1)) - 1) | 1
                for _ in range(20):
                    if is_probable_prime(p):
                        break
                    p += 2
                else:
                    p = None
                
                if p is not None and p > 10**8 and p <= 10**12:
                    candidates.append(p - rng.randint(0, 50))
            
            # Strategy 3: p^2 (phi(p^2) = p*(p-1))
            else:
                bits = rng.randint(16, 28)
                p = 1 << (bits - 1) | rng.randint(0, (1 << (bits - 1)) - 1) | 1
                for _ in range(20):
                    if is_probable_prime(p):
                        break
                    p += 2
                else:
                    p = None
                
                if p is not None:
                    n = p * p
                    if n <= 10**12 and n >= 10**8:
                        candidates.append(n - rng.randint(0, 50))
        
        # Deduplicate and sort
        candidates = list(set(candidates))
        candidates = [c for c in candidates if 1 <= c <= 10**12]
        rng.shuffle(candidates)
        
        return candidates

    best_k = 1
    best_s = 1
    
    # Phase 1: Quick scan of a moderate contiguous range to establish baseline
    W_scan = 500000
    scan_start = 10**6
    
    try:
        phi_vals = compute_phi_segment(scan_start, scan_start + W_scan - 1)
        k, s_rel = find_longest_run(phi_vals)
        actual_s = scan_start + s_rel
        if k > best_k:
            best_k = k
            best_s = actual_s
    except Exception:
        pass

    # Phase 2: Generate and test candidate starts
    if time.time() - start_time < deadline - 10:
        num_candidates = 50
        candidates = generate_candidate_starts(num_candidates, seed=42)
        
        # Window size to check around each candidate
        window = 250  # Check 250 numbers around each candidate
        
        for cand in candidates:
            if time.time() - start_time > deadline - 5:
                break
            
            # Check a window around the candidate
            seg_start = max(1, cand - 50)
            seg_end = min(10**12, seg_start + window - 1)
            
            if seg_start > seg_end:
                continue
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                if best_k >= 200:
                    break
                    
            except Exception:
                continue
    
    # Phase 3: If still not good enough, do a broader contiguous scan
    if best_k < 150 and time.time() - start_time < deadline - 15:
        W_broad = 2000000
        # Scan a few different regions
        regions = [10**8, 10**9, 10**10]
        
        for region_start in regions:
            if time.time() - start_time > deadline - 5:
                break
            
            seg_start = region_start
            seg_end = min(10**12, seg_start + W_broad - 1)
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                if best_k >= 200:
                    break
                    
            except Exception:
                continue

    # Ensure constraints
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
