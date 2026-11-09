import time
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Step 1: Sieve primes up to 10^6 (sqrt(10^12))
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segment(L, R):
        """Compute phi(n) for all n in [L, R] using segmented sieve."""
        n = R - L + 1
        phi = list(range(L, R + 1))  # phi[i] starts as L+i
        is_composite = bytearray(n)
        
        for p in primes:
            if p * p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, R + 1, p):
                idx = j - L
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                is_composite[idx] = 1
        
        # For remaining values that are prime (or 1), phi(p) = p-1
        for i in range(n):
            val = L + i
            if val == 1:
                phi[i] = 1
            elif is_composite[i] == 0 and val > 1:
                # val is prime
                phi[i] = val - 1
        
        return phi
    
    def find_longest_run(phi_vals, L):
        """Find the longest run of consecutive distinct phi values."""
        n = len(phi_vals)
        best_len = 0
        best_start = 0
        seen = set()
        current_start = 0
        
        for i in range(n):
            if phi_vals[i] in seen:
                # Reset
                seen = set()
                current_start = i
            seen.add(phi_vals[i])
            run_len = i - current_start + 1
            if run_len > best_len:
                best_len = run_len
                best_start = current_start
        
        return best_start + L, best_len
    
    # Strategy: Try multiple segments at different scales
    # Start with smaller segments to get a quick result, then try larger
    best_result = (1, 1)  # (length, start)
    
    # Try segments of various sizes
    # For each segment size, try a few starting positions
    segment_sizes = [10000, 50000, 100000, 200000, 500000, 1000000]
    
    for seg_size in segment_sizes:
        if time.time() > deadline - 30:
            break
        
        # Try a few starting positions for this segment size
        # Use positions that are likely to have interesting structure
        # e.g., near powers, or random-ish positions
        num_starts = 5
        for trial in range(num_starts):
            if time.time() > deadline - 15:
                break
            
            # Choose starting position
            if trial == 0:
                L = 1
            elif trial == 1:
                L = 10**6
            elif trial == 2:
                L = 10**8
            elif trial == 3:
                L = 10**10
            else:
                L = (trial * 7 + 3) * 10**7
            
            R = L + seg_size - 1
            if R > 10**12:
                R = 10**12
            if R < L:
                continue
            
            # Compute phi for this segment
            phi_vals = compute_phi_segment(L, R)
            
            # Find longest run
            run_start, run_len = find_longest_run(phi_vals, L)
            
            if run_len > best_result[0]:
                best_result = (run_len, run_start)
    
    return {'start': best_result[1], 'length': best_result[0]}
