"""Long runs of consecutive integers with distinct totients (up to 10^12)."""
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # --- Generate primes up to 10^6 for segmented sieve ---
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # --- Efficient Segmented Phi Computation ---
    def segmented_phi(start, length):
        """Compute phi(n) for n in [start, start+length-1] efficiently."""
        # rem[i] holds the remaining part of (start+i) after dividing by small primes
        rem = list(range(start, start + length))
        phi = list(range(start, start + length))
        
        # Process small primes
        for p in primes:
            if p * p > start + length - 1:
                # If p^2 > max(n), then any remaining factor is prime.
                # But we still need to process p if it divides n.
                # Actually, we should process all primes p <= sqrt(max_n) = 10^6.
                # Since our primes list goes up to 10^6, we just iterate through all of them
                # that are <= start+length-1.
                pass
            
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            for m in range(first, start + length, p):
                idx = m - start
                # Apply phi factor (1 - 1/p)
                phi[idx] = phi[idx] // p * (p - 1)
                # Remove all factors of p from rem[idx]
                while rem[idx] % p == 0:
                    rem[idx] //= p
        
        # Process remaining large prime factors
        for i in range(length):
            if rem[i] > 1:
                # rem[i] is a prime factor > 10^6 (or 1)
                # Since n <= 10^12, rem[i] can be at most 10^12.
                # It must be prime because if it were composite, it would have a factor <= sqrt(rem[i]) <= 10^6,
                # which would have been divided out.
                phi[i] = phi[i] // rem[i] * (rem[i] - 1)
        
        return phi

    # --- Find longest run of distinct phi values ---
    def find_best_run(phi_vals, start_idx):
        """Find the longest run of distinct phi values in phi_vals."""
        best_len = 1
        best_start = start_idx
        seen = set()
        for i in range(len(phi_vals)):
            if phi_vals[i] in seen:
                seen = set()
            seen.add(phi_vals[i])
            # This logic is slightly flawed for finding max contiguous distinct run.
            # Let's use a sliding window with a set.
            
        # Correct sliding window
        best_len = 0
        best_start = start_idx
        seen = set()
        left = 0
        for right in range(len(phi_vals)):
            while phi_vals[right] in seen:
                seen.remove(phi_vals[left])
                left += 1
            seen.add(phi_vals[right])
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = start_idx + left
        return best_len, best_start

    # --- Search Strategy ---
    # We aim for k=200.
    # Random sampling in the large domain is better than sequential small domain.
    # We will sample random starts, check a window of size 200, and if we find a long run,
    # we refine locally.
    
    best_k = 1
    best_s = 1
    window_size = 200
    
    # Helper to evaluate a specific start
    def evaluate_start(s):
        nonlocal best_k, best_s
        if s < 1 or s + window_size - 1 > 10**12:
            return
        phi_vals = segmented_phi(s, window_size)
        k, local_s = find_best_run(phi_vals, s)
        if k > best_k:
            best_k = k
            best_s = local_s

    # Phase 1: Random sampling in [10^9, 10^12]
    # Why 10^9? Numbers are large enough that phi values are large and distinctness is "generic".
    # Small numbers have more structure (primes, powers) leading to collisions?
    # Actually, phi(p) = p-1 and phi(p+1) can collide.
    # Let's just sample broadly.
    
    num_samples = 5000
    for _ in range(num_samples):
        if time.time() > deadline:
            break
        # Sample start uniformly in [10^8, 10^12 - 200]
        s = random.randint(10**8, 10**12 - window_size)
        evaluate_start(s)
        
    # Phase 2: Local refinement around best found
    if best_k > 50:
        # If we found a decent run, try to extend it or find a better one nearby
        # We can slide the window slightly left and right
        for offset in range(-50, 51, 5):
            if time.time() > deadline:
                break
            new_s = best_s + offset
            if new_s >= 1 and new_s + window_size - 1 <= 10**12:
                evaluate_start(new_s)
                
    # Phase 3: If still low, try some specific "safe" structures?
    # Or just more random samples if time permits.
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_k < 100:
        # Do more random samples
        for _ in range(2000):
            if time.time() > deadline:
                break
            s = random.randint(1, 10**12 - window_size)
            evaluate_start(s)

    # Final check and constraint enforcement
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
        if best_k < 1:
            best_s = 1
            best_k = 1
    if best_k > 200:
        best_k = 200
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
