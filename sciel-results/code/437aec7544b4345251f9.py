"""Long runs of consecutive integers with distinct totients (up to 10^12)."""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # --- Generate primes up to 10^6 for segmented sieve ---
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # --- Segmented phi computation over a window ---
    def segmented_phi(start, length):
        """Compute phi(n) for n in [start, start+length-1]."""
        phi = list(range(start, start + length))
        # For each prime p <= limit, mark multiples in the window
        for p in primes:
            if p > start + length - 1:
                break
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            for m in range(first, start + length, p):
                idx = m - start
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        # Handle any remaining prime factor > limit
        # If phi[i] > 1 and the original number had a prime factor > limit,
        # we need to account for it. But since we only have primes up to 10^6,
        # numbers up to 10^12 could have a prime factor > 10^6.
        # Actually, for the segmented approach, we need to be careful.
        # Let's use a different approach: compute phi directly using the formula
        # phi(n) = n * product(1 - 1/p) for each distinct prime p dividing n.
        # For large numbers, we can factor them individually if needed.
        
        # Actually, the segmented sieve above only works correctly if all prime
        # factors are <= limit. For numbers up to 10^12, a number could be the
        # product of two primes both > 10^6, or have a prime factor > 10^6.
        # In that case, the segmented sieve won't fully factor it.
        # 
        # Let's use a hybrid approach: for each number in the window, if after
        # processing all primes <= limit, the remaining value > 1, then it's
        # either 1 or a prime > limit (since if it were composite, it would
        # have a factor <= sqrt(remaining) <= sqrt(10^12) = 10^6, which we've
        # already processed).
        
        for i in range(length):
            n = start + i
            # Recompute phi properly
            # The segmented approach above is actually correct for the part
            # where we subtract phi[idx] // p when p divides the original n.
            # But we need to be more careful. Let me redo this.
            
        # Let me use a cleaner approach
        phi = list(range(start, start + length))
        for p in primes:
            if p > start + length - 1:
                break
            first = ((start + p - 1) // p) * p
            for m in range(first, start + length, p):
                idx = m - start
                phi[idx] = phi[idx] // p * (p - 1)
        
        # Now, for each number, check if there's a remaining prime factor > limit
        for i in range(length):
            n = start + i
            # We need to check if n has a prime factor > limit
            # After dividing by all primes <= limit, what remains?
            # Actually, the phi computation above is correct IF we process all
            # prime factors. The issue is that we might miss a large prime factor.
            # 
            # Let's verify: phi(n) = n * prod_{p|n}(1-1/p)
            # In the segmented approach, for each p <= limit that divides n,
            # we do phi[idx] = phi[idx] // p * (p-1), which multiplies by (1-1/p).
            # If n has a prime factor q > limit, we haven't multiplied by (1-1/q).
            # So we need to detect and handle this.
            
            # Check if n has a prime factor > limit
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                while temp % p == 0:
                    temp //= p
            if temp > 1:
                # temp is a prime factor > limit (or 1)
                # But wait, we already processed all primes <= limit in the phi
                # computation above. So if temp > 1, it's a prime > limit.
                # We need to multiply phi[i] by (1 - 1/temp)
                phi[i] = phi[i] // temp * (temp - 1)
        
        return phi
    
    # --- Find longest run of distinct phi values ---
    def find_best_run(phi_vals, start_idx):
        """Find the longest run of distinct phi values in phi_vals."""
        best_len = 1
        best_start = start_idx
        for i in range(len(phi_vals)):
            seen = set()
            seen.add(phi_vals[i])
            j = i + 1
            while j < len(phi_vals) and phi_vals[j] not in seen:
                seen.add(phi_vals[j])
                j += 1
            run_len = j - i
            if run_len > best_len:
                best_len = run_len
                best_start = i
        return best_len, best_start
    
    # --- Search strategy ---
    # Start with small numbers, then expand to larger ranges
    best_k = 1
    best_s = 1
    
    # Phase 1: Small numbers (1 to 100000)
    window_size = 10000
    for base in range(1, 100001, window_size):
        if time.time() > deadline:
            break
        end = min(base + window_size, 100001)
        length = end - base
        phi_vals = segmented_phi(base, length)
        k, s = find_best_run(phi_vals, base)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Larger ranges, sample at strategic points
    # Use larger windows and jump around
    window_size = 50000
    step = 500000  # Jump by 500k each time
    
    # Try ranges around powers and other interesting values
    search_points = [
        10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
        2 * 10**6, 5 * 10**6, 10**7, 2 * 10**7, 5 * 10**7,
        10**8, 2 * 10**8, 5 * 10**8, 10**9, 2 * 10**9, 5 * 10**9,
        10**10, 2 * 10**10, 5 * 10**10, 10**11, 2 * 10**11, 5 * 10**11,
    ]
    
    # Also try consecutive ranges
    for base in range(1, 10**6, step):
        if time.time() > deadline:
            break
        end = min(base + window_size, 10**6)
        length = end - base
        phi_vals = segmented_phi(base, length)
        k, s = find_best_run(phi_vals, base)
        if k > best_k:
            best_k = k
            best_s = s
    
    for pt in search_points:
        if time.time() > deadline:
            break
        if pt + window_size > 10**12:
            continue
        phi_vals = segmented_phi(pt, window_size)
        k, s = find_best_run(phi_vals, pt)
        if k > best_k:
            best_k = k
            best_s = s
        # Also try pt - window_size
        if pt - window_size >= 1:
            phi_vals = segmented_phi(pt - window_size, window_size)
            k, s = find_best_run(phi_vals, pt - window_size)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 3: Finer search around best found so far
    if best_k > 10:
        if time.time() < deadline - 30:
            # Search around best_s with smaller steps
            search_start = max(1, best_s - 100000)
            search_end = min(10**12, best_s + 100000)
            for base in range(search_start, search_end, 10000):
                if time.time() > deadline:
                    break
                end = min(base + window_size, search_end)
                length = end - base
                if length < 2:
                    continue
                phi_vals = segmented_phi(base, length)
                k, s = find_best_run(phi_vals, base)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Ensure constraints are met
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
