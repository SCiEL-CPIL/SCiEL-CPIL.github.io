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

    def compute_phi_segment(start, end):
        """
        Computes phi(n) for n in [start, end] using segmented sieve.
        Returns a numpy array of phi values (int64).
        """
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        # Use int64 to be safe, though phi(n) < n <= 10^12 fits in int64
        vals = np.arange(start, end + 1, dtype=np.int64)
        phis = vals.copy()
        rems = vals.copy()
        
        # Process small prime factors
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            if first_mult > end:
                continue
                
            # Use direct index computation instead of np.arange
            # Number of multiples: floor((end - first_mult) / p) + 1
            count = (end - first_mult) // p + 1
            if count <= 0:
                continue
            
            # Indices of multiples of p in [start, end]
            idx_array = np.arange(first_mult - start, size, p, dtype=np.int64)
            
            # Only process where rems is actually divisible by p
            mask = rems[idx_array] % p == 0
            valid_idx = idx_array[mask]
            
            if len(valid_idx) == 0:
                continue
            
            # Update phi: phi[n] -= phi[n] // p
            phis[valid_idx] -= phis[valid_idx] // p
            
            # Remove all factors of p from rems - do it in one pass
            # For each valid_idx, divide rems by p as many times as possible
            # Use a loop but with early exit
            for _ in range(10):  # p^10 > 10^12 for p >= 2
                still_div = rems[valid_idx] % p == 0
                if not np.any(still_div):
                    break
                rems[valid_idx[still_div]] //= p
                    
        # Process remaining large prime factors
        # For most numbers, rems[i] will be 1 or a prime > 10^6
        # We need to handle the case where rems[i] > 1
        # Since p > 10^6, p^2 > 10^12, so rems[i] can only be 1 or a prime.
        
        # Find indices where rems > 1
        large_prime_mask = rems > 1
        if np.any(large_prime_mask):
            large_indices = np.where(large_prime_mask)[0]
            p_vals = rems[large_indices]
            phis[large_indices] -= phis[large_indices] // p_vals
            
        return phis

    def find_longest_run(phi_vals, start_index):
        """
        Finds the longest run of distinct values in phi_vals.
        phi_vals is a numpy array.
        Returns (max_length, relative_start_index).
        """
        best_k = 1
        best_s_rel = 0
        n = len(phi_vals)
        
        if n == 0:
            return 0, 0

        # Use a dictionary to track last seen index of each value
        last_seen = {}
        current_len = 0
        current_start_rel = 0
        
        for i in range(n):
            val = int(phi_vals[i])
            
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

    def check_run_distinct(start, k):
        """
        Check if phi(start), phi(start+1), ..., phi(start+k-1) are all distinct.
        Returns True if distinct, False otherwise.
        """
        if start < 1 or start + k - 1 > 10**12:
            return False
        if k <= 1:
            return True
        
        # Compute phi for the range
        phi_vals = compute_phi_segment(start, start + k - 1)
        if len(phi_vals) != k:
            return False
        
        # Check for duplicates
        seen = set()
        for val in phi_vals:
            v = int(val)
            if v in seen:
                return False
            seen.add(v)
        return True

    def extend_run(start, initial_k):
        """
        Try to extend a run starting at 'start' beyond initial_k.
        Returns the maximum k such that phi(start), ..., phi(start+k-1) are distinct.
        """
        k = initial_k
        while k < 200 and start + k <= 10**12:
            if check_run_distinct(start, k + 1):
                k += 1
            else:
                break
        return k

    def inverse_totient_candidates(y, max_candidates=10):
        """
        Generate candidate s such that phi(s) is close to y.
        This is an approximation - we look for s where phi(s) ~ y.
        For s = p*q where p, q are primes, phi(s) = (p-1)(q-1).
        We can try to factor y and find such pairs.
        """
        candidates = set()
        
        # Try s = 2*y + 2 (since phi(2p) = phi(p) = p-1 for prime p, so if y = p-1, then s = 2p = 2(y+1))
        p_candidate = y + 1
        if p_candidate > 2 and is_prime[min(p_candidate, limit - 1) if p_candidate <= limit else 0] == 1:
            # This is a rough check - for large p_candidate we need better primality
            # For now, just add as candidate
            candidates.add(2 * p_candidate)
        
        # Try s = y + 1 (if y+1 is prime, phi(y+1) = y)
        if y + 1 > 2:
            candidates.add(y + 1)
        
        # Try s = 2*(y+1) (if y+1 is prime, phi(2*(y+1)) = phi(y+1) = y)
        candidates.add(2 * (y + 1))
        
        # Try to find s where s has a large prime factor
        # If s = p * m where p is prime > sqrt(s), then phi(s) = (p-1) * phi(m)
        # So we need (p-1) * phi(m) ≈ y
        # Try small m values
        for m in range(2, 20):
            if y % m == 0:
                # phi(m) must divide y/m... this is complex
                # Just add s = m * (y//m + 1) as a rough candidate
                s_candidate = m * (y // m + 1)
                if s_candidate >= 1 and s_candidate <= 10**12:
                    candidates.add(s_candidate)
        
        return list(candidates)[:max_candidates]

    best_k = 1
    best_s = 1
    
    # Strategy 1: Inverse totient candidate generation
    # Generate candidates by solving for s where phi(s) ≈ y for random y
    if time.time() - start_time < deadline - 30:
        random.seed(42)
        
        # Sample random y values in high range
        for trial in range(200):
            if time.time() - start_time > deadline - 20:
                break
            
            # y in [10^11, 10^12]
            y = random.randint(10**11, 10**12 - 1000)
            
            # Generate candidates
            candidates = inverse_totient_candidates(y, max_candidates=20)
            
            for s_candidate in candidates:
                if time.time() - start_time > deadline - 20:
                    break
                
                # Ensure s_candidate is in valid range
                if s_candidate < 1 or s_candidate > 10**12 - 200:
                    continue
                
                # Check if this starting point gives a long run
                # First check a window of 200
                window_end = min(s_candidate + 199, 10**12)
                window_size = window_end - s_candidate + 1
                
                if window_size < 10:  # Too small, skip
                    continue
                
                try:
                    phi_vals = compute_phi_segment(s_candidate, window_end)
                    k, s_rel = find_longest_run(phi_vals, s_candidate)
                    actual_s = s_candidate + s_rel
                    
                    if k > best_k:
                        best_k = k
                        best_s = actual_s
                        
                        # Try to extend the run
                        if time.time() - start_time < deadline - 15:
                            extended_k = extend_run(actual_s, k)
                            if extended_k > best_k:
                                best_k = extended_k
                                best_s = actual_s
                            
                    if best_k >= 200:
                        break
                except Exception:
                    continue
            
            if best_k >= 200:
                break
    
    # Strategy 2: Scan contiguous regions in high ranges where totients are sparser
    # For large n, phi(n) is typically much smaller than n, reducing collision probability
    
    W = 200000  # 200k numbers per segment - smaller for more segments
    
    # Scan several segments in high ranges - focus on ranges likely to have long runs
    scan_ranges = [
        (8 * 10**11, 8 * 10**11 + W * 5),
        (9 * 10**11, 9 * 10**11 + W * 5),
        (10**12 - W * 5, 10**12),
    ]
    
    for range_start, range_end in scan_ranges:
        if time.time() - start_time > deadline - 5:
            break
        
        for seg_start in range(range_start, range_end, W):
            if time.time() - start_time > deadline - 5:
                break
            
            seg_end = min(seg_start + W - 1, range_end, 10**12)
            if seg_start > seg_end:
                break
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals, seg_start)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    # Try to extend the run
                    if time.time() - start_time < deadline - 10:
                        extended_k = extend_run(actual_s, k)
                        if extended_k > best_k:
                            best_k = extended_k
                            best_s = actual_s
                            
                if best_k >= 200:
                    return {'start': best_s, 'length': best_k}
                    
            except Exception:
                continue

    # Strategy 3: Random sampling in high ranges
    if time.time() - start_time < deadline - 10:
        random.seed(123)
        
        # Sample random starting points in very high ranges
        for _ in range(30):
            if time.time() - start_time > deadline - 5:
                break
            
            # Focus on ranges where n is large
            seg_start = random.randint(7 * 10**11, 10**12 - W)
            seg_end = seg_start + W - 1
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals, seg_start)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    # Try to extend
                    if time.time() - start_time < deadline - 5:
                        extended_k = extend_run(actual_s, k)
                        if extended_k > best_k:
                            best_k = extended_k
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
