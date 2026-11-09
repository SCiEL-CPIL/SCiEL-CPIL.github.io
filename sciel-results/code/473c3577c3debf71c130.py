import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Step 1: Get primes up to sqrt(10^12) = 10^6 using sieve of Eratosthenes
    limit = 10**6
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0]
    primes_list = primes.tolist()
    
    def compute_phi_segment(lo, hi, primes_arr):
        """Compute phi(n) for n in [lo, hi] using segmented sieve."""
        n = hi - lo + 1
        phi_vals = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes_arr:
            p = int(p)
            if p * p > hi:
                break
            first = ((lo + p - 1) // p) * p
            if first < lo:
                first += p
            if first > hi:
                continue
            # Vectorized: find all indices where remaining is divisible by p
            for j in range(first, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors
        for i in range(n):
            if remaining[i] > 1:
                phi_vals[i] = phi_vals[i] // remaining[i] * (remaining[i] - 1)
        
        return phi_vals
    
    def find_best_run(phi_arr, base_start):
        """Find longest run of distinct phi values using sliding window."""
        best_k_local = 1
        best_s_local = base_start
        count = {}
        left = 0
        for right in range(len(phi_arr)):
            val = int(phi_arr[right])
            count[val] = count.get(val, 0) + 1
            while count[val] > 1:
                left_val = int(phi_arr[left])
                count[left_val] -= 1
                if count[left_val] == 0:
                    del count[left_val]
                left += 1
            current_k = right - left + 1
            if current_k > best_k_local:
                best_k_local = current_k
                best_s_local = base_start + left
        return best_k_local, best_s_local
    
    # Phase 1: Scan small range [1, 1000000]
    small_hi = 1000000
    phi_small = compute_phi_segment(1, small_hi, primes_list)
    best_k, best_s = find_best_run(phi_small, 1)
    
    # Phase 2: Scan medium ranges [10^6, 10^8] in segments
    seg_size = 200000
    search_start = small_hi + 1
    max_search = 10**8
    
    while search_start < max_search and time.time() < deadline - 5:
        seg_end = min(search_start + seg_size - 1, max_search)
        if time.time() > deadline - 10:
            break
        phi_seg = compute_phi_segment(search_start, seg_end, primes_list)
        k_local, s_local = find_best_run(phi_seg, search_start)
        if k_local > best_k:
            best_k = k_local
            best_s = s_local
        search_start = seg_end + 1
        if best_k >= 200:
            break
    
    # Phase 3: If still not enough, try larger ranges with sparser sampling
    if best_k < 100 and time.time() < deadline - 10:
        seg_size = 100000
        # Try ranges at larger bases
        bases = list(range(10**8, 10**10, 10**7))
        for base in bases:
            if time.time() > deadline - 15:
                break
            seg_start = base
            seg_end = base + seg_size - 1
            if seg_end > 10**12:
                break
            phi_seg = compute_phi_segment(seg_start, seg_end, primes_list)
            k_local, s_local = find_best_run(phi_seg, seg_start)
            if k_local > best_k:
                best_k = k_local
                best_s = s_local
            if best_k >= 200:
                break
    
    # Phase 4: Final check - try very large bases if time permits
    if best_k < 150 and time.time() < deadline - 10:
        seg_size = 50000
        # Try some very large random-ish bases
        import random
        random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            base = random.randint(10**10, 10**12 - seg_size)
            seg_start = base
            seg_end = base + seg_size - 1
            phi_seg = compute_phi_segment(seg_start, seg_end, primes_list)
            k_local, s_local = find_best_run(phi_seg, seg_start)
            if k_local > best_k:
                best_k = k_local
                best_s = s_local
            if best_k >= 200:
                break
    
    # Ensure we return a valid result with Python ints
    best_k = int(min(best_k, 200))
    best_s = int(best_s)
    
    return {'start': best_s, 'length': best_k}
