"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Segmented sieve for phi computation over arbitrary ranges
    def compute_phi_segmented(start, end):
        """Compute phi for all numbers in [start, end] using segmented sieve."""
        size = end - start + 1
        if size <= 0:
            return []
        
        # Initialize phi values
        phi = list(range(start, end + 1))
        remaining = list(range(start, end + 1))
        
        # Sieve with primes up to sqrt(end)
        limit = int(math.isqrt(end)) + 1
        for p in range(2, limit + 1):
            # Check if p is prime by seeing if remaining[p-start] hasn't been modified
            if p <= end and remaining[p - start] == p:
                # p is prime, process multiples
                first_mult = ((start + p - 1) // p) * p
                if first_mult < p * p:
                    first_mult = p * p
                for m in range(first_mult, end + 1, p):
                    idx = m - start
                    if remaining[idx] % p == 0:
                        phi[idx] -= phi[idx] // p
                        while remaining[idx] % p == 0:
                            remaining[idx] //= p
        
        # Handle remaining prime factors
        for i in range(size):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    # Find longest run of distinct phi values using sliding window
    def find_best_run_sliding(phi_values, start_idx, end_idx):
        """Find the longest run of distinct phi values using sliding window."""
        if end_idx - start_idx < 1:
            return 1, start_idx
        
        best_k = 1
        best_start = start_idx
        window_start = start_idx
        seen = {}
        
        for i in range(start_idx, min(end_idx, len(phi_values))):
            val = phi_values[i]
            if val in seen and seen[val] >= window_start:
                # Move window start past the previous occurrence
                window_start = seen[val] + 1
            seen[val] = i
            
            current_len = i - window_start + 1
            if current_len > best_k:
                best_k = current_len
                best_start = window_start
        
        return best_k, best_start
    
    # Precompute small primes for segmented sieve
    def get_primes_up_to(n):
        """Sieve of Eratosthenes for primes up to n."""
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    best_k = 1
    best_start = 1
    
    # Phase 1: Search small ranges with full sieve
    small_limits = [10**4, 10**5, 10**6, 10**7]
    for limit in small_limits:
        if time_remaining() < 5.0:
            break
        try:
            phi_vals = compute_phi_segmented(1, limit)
            k, s = find_best_run_sliding(phi_vals, 1, limit + 1)
            if k > best_k:
                best_k = k
                best_start = s
        except Exception:
            continue
    
    # Phase 2: Search larger ranges using segmented sieve in strategic locations
    if time_remaining() > 10.0 and best_k < 200:
        # Strategy: search around numbers with many distinct prime factors
        # These tend to produce diverse phi values
        search_regions = []
        
        # Region around highly composite-like numbers
        for base in [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]:
            if time_remaining() < 5.0:
                break
            # Search a window of size 5000 around base
            window_size = 5000
            start = max(1, base - window_size // 2)
            end = min(10**12, base + window_size // 2)
            try:
                phi_vals = compute_phi_segmented(start, end)
                k, s = find_best_run_sliding(phi_vals, 0, len(phi_vals))
                if k > best_k:
                    best_k = k
                    best_start = start + s
            except Exception:
                continue
        
        # Random search in large ranges
        random.seed(42)
        for _ in range(50):
            if time_remaining() < 3.0:
                break
            s = random.randint(1, 10**12 - 200)
            window_size = 200
            try:
                phi_vals = compute_phi_segmented(s, s + window_size - 1)
                k, offset = find_best_run_sliding(phi_vals, 0, len(phi_vals))
                if k > best_k:
                    best_k = k
                    best_start = s + offset
            except Exception:
                continue
    
    # Ensure constraints are met
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    
    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
