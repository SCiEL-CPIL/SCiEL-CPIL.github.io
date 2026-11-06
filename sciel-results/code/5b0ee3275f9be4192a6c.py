import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    # Sieve primes up to 10^6
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes = sieve_primes(1000000)

    def phi_single(n):
        """Compute phi(n) using precomputed primes."""
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result

    def compute_phi_segment(start, length):
        """Compute phi for consecutive integers [start, start+length-1] using segmented sieve."""
        if length <= 0:
            return []
        # Initialize phi values
        phi_vals = list(range(start, start + length))
        
        # For each prime, update phi for its multiples in the range
        for p in primes:
            if p > start + length - 1:
                break
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            # Update phi for all multiples of p in range
            for m in range(first, start + length, p):
                idx = m - start
                phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle remaining large prime factors
        # For each number, if after removing all small prime factors there's a remainder > 1,
        # that remainder is a prime factor
        for i in range(length):
            n = start + i
            temp = n
            # Remove all small prime factors
            for p in primes:
                if p * p > temp:
                    break
                while temp % p == 0:
                    temp //= p
            if temp > 1:
                phi_vals[i] -= phi_vals[i] // temp
        
        return phi_vals

    def find_longest_distinct_run(phi_vals, offset, max_k=200):
        """Find the longest run of pairwise distinct phi values."""
        best_k = 1
        best_pos = 0
        n = len(phi_vals)
        
        for s in range(n):
            if time.time() > deadline - 5:
                break
            seen = set()
            k = 0
            while s + k < n and s + k - s < max_k:
                if phi_vals[s + k] in seen:
                    break
                seen.add(phi_vals[s + k])
                k += 1
            if k > best_k:
                best_k = k
                best_pos = s
                if best_k == max_k:
                    return best_k, offset + best_pos
        return best_k, offset + best_pos

    best_k = 1
    best_s = 1

    # Phase 1: Try near 10^12 with large segments
    # Compute phi for a large window near 10^12
    window_size = 50000
    near_top = 10**12 - window_size + 1
    if near_top >= 1:
        phi_top = compute_phi_segment(near_top, window_size)
        k, s = find_longest_distinct_run(phi_top, near_top)
        if k > best_k:
            best_k, best_s = k, s

    # Phase 2: Try various strategic positions
    # Small numbers
    if time.time() < deadline - 20:
        phi_small = compute_phi_segment(1, 100000)
        k, s = find_longest_distinct_run(phi_small, 1)
        if k > best_k:
            best_k, best_s = k, s

    # Medium positions
    medium_starts = [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]
    for ms in medium_starts:
        if time.time() > deadline - 30:
            break
        if ms + window_size - 1 > 10**12:
            continue
        phi_med = compute_phi_segment(ms, window_size)
        k, s = find_longest_distinct_run(phi_med, ms)
        if k > best_k:
            best_k, best_s = k, s

    # Phase 3: If we haven't found a long run, try more positions
    if best_k < 100 and time.time() < deadline - 10:
        random.seed(123)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            s_rand = random.randint(1, 10**12 - window_size + 1)
            phi_rand = compute_phi_segment(s_rand, window_size)
            k, s = find_longest_distinct_run(phi_rand, s_rand)
            if k > best_k:
                best_k, best_s = k, s

    # Phase 4: If we found a good run, try to extend it by checking nearby
    if best_k >= 50 and time.time() < deadline - 5:
        # Check if we can find an even longer run by looking at overlapping windows
        # Try positions just before and after best_s
        for delta in [-window_size//2, -window_size//4, window_size//4, window_size//2]:
            if time.time() > deadline - 3:
                break
            s_check = best_s + delta
            if s_check < 1 or s_check + window_size - 1 > 10**12:
                continue
            phi_check = compute_phi_segment(s_check, window_size)
            k, s = find_longest_distinct_run(phi_check, s_check)
            if k > best_k:
                best_k, best_s = k, s

    # Final verification and return
    if best_k > 200:
        best_k = 200
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
        if best_k < 1:
            best_s = 1
            best_k = 1

    return {'start': best_s, 'length': best_k}
