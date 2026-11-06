"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Precompute small primes up to sqrt(10^12) = 10^6
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]
    
    primes_small = sieve_primes(10**6 + 1)
    
    # Segmented phi sieve using NumPy
    def compute_phi_segmented(start, end):
        """Compute phi for numbers in [start, end] using segmented approach."""
        n = end - start + 1
        phi = np.arange(start, end + 1, dtype=np.int64)
        
        for p in primes_small:
            if p > end:
                break
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            # For all multiples of p, phi[x] -= phi[x] // p
            # Use vectorized operation
            indices = np.arange(first, end + 1, p)
            local_indices = indices - start
            phi[local_indices] -= phi[local_indices] // p
        
        return phi
    
    # Fast phi for individual large numbers using precomputed primes
    def phi_large(x):
        """Compute phi for a single large number using precomputed primes."""
        result = x
        y = x
        for p in primes_small:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    # Find longest run of distinct phi values using sliding window
    def find_best_run(phi_values, start_idx, end_idx):
        """Find the longest run of distinct phi values using O(N) sliding window."""
        n = len(phi_values)
        if n == 0:
            return 1, start_idx
        
        best_k = 1
        best_start = start_idx
        
        # Sliding window approach
        from collections import defaultdict
        count = defaultdict(int)
        left = 0
        current_max = 1
        
        for right in range(n):
            val = phi_values[right]
            count[val] += 1
            
            while count[val] > 1:
                left_val = phi_values[left]
                count[left_val] -= 1
                left += 1
            
            window_len = right - left + 1
            if window_len > current_max:
                current_max = window_len
                best_k = window_len
                best_start = start_idx + left
        
        return min(best_k, 200), best_start
    
    # Strategy 1: Search in large contiguous ranges using segmented sieve
    best_k = 1
    best_start = 1
    
    # Try increasing segment sizes
    segment_sizes = [10**5, 2*10**5, 5*10**5, 10**6, 2*10**6, 5*10**6, 10**7, 2*10**7, 5*10**7]
    
    for size in segment_sizes:
        if time_remaining() < 5.0:
            break
        
        try:
            # Search multiple segments of this size
            max_search = 10**12
            num_segments = min(10, int(time_remaining() / 2))
            
            for seg_idx in range(num_segments):
                if time_remaining() < 2.0:
                    break
                
                # Spread segments across the range
                base = seg_idx * (max_search // num_segments)
                seg_start = base + 1
                seg_end = min(base + size, max_search)
                
                if seg_end - seg_start + 1 < 100:
                    continue
                
                phi_vals = compute_phi_segmented(seg_start, seg_end)
                k, s = find_best_run(phi_vals, seg_start, seg_end)
                
                if k > best_k:
                    best_k = k
                    best_start = s
                    if best_k >= 200:
                        return {'start': best_start, 'length': best_k}
                        
        except MemoryError:
            continue
        except Exception:
            continue
    
    # Strategy 2: Targeted search in random large ranges
    if time_remaining() > 5.0 and best_k < 200:
        import random
        random.seed(42)
        max_search = 10**12
        
        for _ in range(50):
            if time_remaining() < 2.0:
                break
            
            s = random.randint(1, max_search - 200)
            
            # Check a window of size up to 200
            seen = set()
            k = 0
            for i in range(200):
                if s + i > max_search:
                    break
                val = phi_large(s + i)
                if val not in seen:
                    seen.add(val)
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_start = s
                if best_k >= 200:
                    break
    
    # Strategy 3: Try structured ranges (near multiples of primorials, etc.)
    if time_remaining() > 3.0 and best_k < 200:
        # Try ranges near numbers with many small prime factors
        structured_starts = [
            30030 + 1, 210 + 1, 2310 + 1, 30030 * 10 + 1,
            10**6 + 1, 10**7 + 1, 10**8 + 1, 10**9 + 1,
            999999937 + 1, 999999999989 + 1
        ]
        
        for s in structured_starts:
            if time_remaining() < 1.0:
                break
            if s + 200 > 10**12:
                continue
            
            seen = set()
            k = 0
            for i in range(200):
                if s + i > 10**12:
                    break
                val = phi_large(s + i)
                if val not in seen:
                    seen.add(val)
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_start = s
                if best_k >= 200:
                    break
    
    # Ensure constraints are met
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = max(1, 10**12 - best_k + 1)
    if best_start < 1:
        best_start = 1
    
    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
