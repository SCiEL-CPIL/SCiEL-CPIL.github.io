import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to a reasonable limit for fast factorization
    # We need primes up to sqrt(10^12) = 10^6 for trial division fallback,
    # but for segmented phi, we only need primes up to sqrt(max_n) for the segment.
    # Actually, for phi(n) we need all prime factors of n.
    # If we use a segmented sieve approach for phi, we need primes up to sqrt(max_value_in_segment).
    # But max_value can be 10^12, so sqrt is 10^6.
    # Let's sieve primes up to 10^6.
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, length):
        """
        Compute phi(x) for x in [start, start+length-1].
        Uses segmented sieve: for each prime p, mark multiples in the range.
        """
        if length <= 0:
            return []
        # phi values initialized to x
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        # temp array to track if already processed? No, standard segmented phi:
        # phi(x) = x * product(1 - 1/p) for distinct prime factors p.
        # We can do: result = x; for each p dividing x, result -= result // p.
        # In segmented sieve:
        # For each prime p:
        #   find first multiple of p in [start, start+length)
        #   for each multiple m, if m % p == 0:
        #       phi[m - start] -= phi[m - start] // p
        # But we must ensure we only do this once per distinct prime factor.
        # Since we iterate primes in increasing order, and we divide out p completely?
        # No, the formula phi(n) = n * Π(1-1/p) works if we apply it for each distinct p.
        # So: for each p, for each multiple m of p in range:
        #   phi[m] = phi[m] // p * (p-1)  -- this is equivalent to phi[m] -= phi[m]//p
        # But wait: if m is divisible by p^2, we still only apply the factor (1-1/p) once.
        # So yes, for each distinct prime p, apply the update once for each multiple.
        
        for p in primes:
            if p * p > start + length - 1:
                # p is larger than sqrt(max), so if p divides any number in range,
                # that number can have at most one such large prime factor.
                # But we still need to handle it.
                # Actually, the loop condition should just be: process all primes p <= start+length-1?
                # No, that's too many.
                # Standard segmented sieve for phi:
                # For primes p <= sqrt(max_val): process normally.
                # For primes p > sqrt(max_val): each number in range can have at most one such factor.
                # We can handle large primes separately or just continue the loop for all primes up to max_val?
                # That's too slow.
                
                # Better: continue processing primes while p <= start + length - 1
                # But we only have primes up to 10^6.
                # If start is large (e.g., 10^12), then p <= 10^6 is much smaller than start.
                # So p*p <= 10^12, which is true for all our primes.
                # So we should process ALL primes in our list.
                pass
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            for m in range(first, start + length, p):
                idx = m - start
                phi_vals[idx] = phi_vals[idx] // p * (p - 1)
        
        # Handle remaining large prime factors?
        # After processing all primes p <= 10^6, any remaining factor of phi_vals[idx] 
        # (if phi_vals[idx] != 1) must be a prime > 10^6.
        # But phi(n) is not n. We are computing phi(n) directly.
        # Wait, the algorithm above computes phi(n) by starting with n and applying the formula.
        # So after processing all small primes, if the current value of phi_vals[idx] is still > 1,
        # it means there is a large prime factor q > 10^6 that divides the original n.
        # But we haven't applied the (1-1/q) factor yet.
        # However, if q > 10^6 and q divides n, then n/q < 10^6 (since n <= 10^12).
        # So we can check if the remaining part is prime.
        
        # Actually, the standard way:
        # After the sieve, for each i, if phi_vals[i] > 1, then phi_vals[i] is a prime factor > 10^6.
        # But phi_vals[i] is the current value of phi(n), not n.
        # This is incorrect. The segmented sieve for phi works on n, not phi(n).
        # Let's re-derive:
        # We want phi(n). We start with val = n.
        # For each prime p dividing n, we do val = val // p * (p-1).
        # So we need to know the distinct prime factors of n.
        # The segmented sieve marks multiples of p. When we process p, we update val for all multiples of p.
        # But if n has a prime factor q > 10^6, we never process q in the loop.
        # So after the loop, val might still be missing the factor (1-1/q).
        # How to detect q? 
        # If n is divisible by q > 10^6, then n/q < 10^6.
        # So after processing all p <= 10^6, if val != 1, it doesn't directly tell us about q.
        # Actually, the value `val` is not n anymore. It's partially reduced.
        
        # Correct approach for segmented phi:
        # 1. Start with arr[i] = start + i.
        # 2. For each prime p in primes:
        #      for each multiple m of p in range:
        #          arr[m-start] = arr[m-start] // p * (p-1)
        # 3. After all primes, for each i:
        #      if arr[i] > 1:
        #          # This means there's a prime factor > 10^6 remaining in the ORIGINAL number?
        #          # No, arr[i] is the current phi-like value.
        #          # Let's trace: n = p*q, p small, q large.
        #          # Start: val = p*q.
        #          # Process p: val = p*q // p * (p-1) = q*(p-1).
        #          # q is not processed.
        #          # Final val = q*(p-1). But phi(n) = (p-1)*(q-1).
        #          # So we are missing the (q-1)/q factor.
        #          # The remaining "large prime" is q.
        #          # How to find q? 
        #          # We know that the original number n = start+i.
        #          # We can compute the remaining factor by checking if n / (product of small prime factors) is prime.
        #          # But we didn't store the product.
        
        # Alternative: Use a different method for large numbers.
        # Given the time limit, maybe we should just use trial division with the precomputed primes for each number?
        # That might be slower but simpler.
        # Or, we can use the fact that if a number has a large prime factor, it's unique.
        # Let's try a hybrid: 
        # For the segmented sieve, after processing all small primes, 
        # for each number n in the range, let rem = n.
        # For each small prime p dividing n, rem //= p (while rem % p == 0).
        # If rem > 1, then rem is a prime factor (since if it were composite, it would have a factor <= sqrt(rem) <= sqrt(n) <= 10^6, which we would have found).
        # So if rem > 1, we apply the factor (1 - 1/rem).
        
        # Let's implement this correctly.
        # We need to track the remaining part of n after removing small prime factors.
        # This requires an extra array or re-computation.
        
        # Given the complexity, let's use a simpler method: 
        # Precompute primes up to 10^6.
        # For each n in the range, factorize n using these primes.
        # If after removing all small prime factors, rem > 1, then rem is a large prime.
        # This is O(k * pi(sqrt(n))) per number, which might be slow for large k.
        # But k is at most 200. And we only do this for a few ranges.
        
        return phi_vals

    def phi_single(n):
        """Compute phi(n) using precomputed primes."""
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    def find_best_run(start, length, max_k=200):
        """Find the longest run of distinct phi values in [start, start+length)."""
        if length <= 0:
            return 0, start
        # Compute phi for the range
        phis = [phi_single(start + i) for i in range(length)]
        best_k = 1
        best_s = start
        for s_idx in range(length - max_k + 1):
            seen = set()
            k = 0
            for i in range(s_idx, s_idx + max_k):
                if phis[i] not in seen:
                    seen.add(phis[i])
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_s = start + s_idx
        return best_k, best_s

    # Strategy:
    # 1. Try a few random ranges near 10^12.
    # 2. Try a few ranges near smaller values where runs might be longer?
    # 3. Use the time budget to search as many ranges as possible.
    
    best_k = 1
    best_s = 1
    
    # First, check a small range to get a baseline
    if time_left() > 5:
        k, s = find_best_run(1, 10000, 200)
        if k > best_k:
            best_k, best_s = k, s

    # Random search in large ranges
    random.seed(42)
    num_tries = 0
    while time_left() > 10:
        # Pick a random start in [10^9, 10^12]
        start = random.randint(10**9, 10**12 - 200)
        length = 1000  # Look in a window of 1000 numbers
        k, s = find_best_run(start, length, 200)
        if k > best_k:
            best_k, best_s = k, s
        num_tries += 1
        if num_tries % 10 == 0:
            # Occasionally try a smaller range for finer search?
            pass

    # Also try to extend the best run found?
    # If best_k < 200, try to extend it left and right.
    if best_k < 200 and time_left() > 5:
        # Try extending left
        left_start = max(1, best_s - 1000)
        left_len = best_s - left_start + best_k + 1000
        k_left, s_left = find_best_run(left_start, left_len, 200)
        if k_left > best_k:
            best_k, best_s = k_left, s_left
        
        # Try extending right
        right_end = min(10**12, best_s + best_k + 1000)
        right_start = best_s
        right_len = right_end - right_start + 1
        k_right, s_right = find_best_run(right_start, right_len, 200)
        if k_right > best_k:
            best_k, best_s = k_right, s_right

    return {'start': best_s, 'length': best_k}
