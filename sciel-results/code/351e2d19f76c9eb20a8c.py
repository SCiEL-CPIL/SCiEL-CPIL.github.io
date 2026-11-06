<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_prime(n):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Deterministic Miller-Rabin for n < 3.3e24
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Bases sufficient for n < 3,317,044,064,679,887,385,961,981
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = 2 + (n - 3) * (hash(str(n)) % max(1, n - 3)) if n > 3 else 2
        c = 1 + (n - 2) * (hash(str(n) + 'c') % max(1, n - 2)) if n > 2 else 1
        m = 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g

def _factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if _is_prime(n):
        factors.append(n)
        return factors
    d = _pollard_brent(n)
    _factor(d, factors)
    _factor(n // d, factors)
    return factors

def _largest_prime_factor(n):
    if n <= 1:
        return 1
    # Trial divide by small primes first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if n % p == 0:
            while n % p == 0:
                n //= p
    if n == 1:
        return max(small_primes[:1])  # fallback, shouldn't happen for n>1
    if _is_prime(n):
        return n
    factors = _factor(n)
    return max(factors)

def _is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # Quick check: if n itself is prime and > p, fail
    if _is_prime(n) and n > p:
        return False
    # Trial divide by primes up to min(p, sqrt(n))
    # But we need to check if ANY prime factor > p exists
    # Strategy: divide out all factors <= p, see what's left
    temp = n
    # First divide by 2
    while temp % 2 == 0:
        temp //= 2
    d = 3
    limit = int(math.isqrt(temp)) + 1
    while d <= limit and d <= p:
        while temp % d == 0:
            temp //= d
            limit = int(math.isqrt(temp)) + 1
        d += 2
    if temp == 1:
        return True
    # temp > 1, check if temp is prime and <= p
    if temp <= p:
        return True
    # If temp > p, it has a prime factor > p (since we divided all factors <= p)
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s, 5s safety margin
    
    # Precompute primes up to some limit for trial division
    # We'll search for primes p and check smoothness of p^2+1 ... p^2+k
    
    best_k = 0
    best_p = 2
    
    # Generate primes using a simple sieve up to some limit, then extend
    # Start with small primes, then check larger ones
    # For efficiency, let's first try a range of primes
    
    # Sieve primes up to 100000 initially
    def sieve(n):
        is_prime_arr = [True] * (n + 1)
        is_prime_arr[0] = is_prime_arr[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime_arr[i]:
                for j in range(i * i, n + 1, i):
                    is_prime_arr[j] = False
        return [i for i in range(2, n + 1) if is_prime_arr[i]]
    
    # Start with primes up to 10000
    primes = sieve(10000)
    
    for p in primes:
        if time.time() > deadline:
            break
        if p <= 1:
            continue
        k = 0
        # Check p^2 + 1, p^2 + 2, ..., p^2 + k
        # We want the longest run where all are p-smooth
        base = p * p
        # We can check up to 200
        for offset in range(1, 201):
            if time.time() > deadline:
                break
            n = base + offset
            if not _is_p_smooth(n, p):
                break
            k = offset
        if k > best_k:
            best_k = k
            best_p = p
            # Early exit if we hit the max
            if best_k >= 200:
                break
    
    # If we didn't find a good one in small primes, try larger primes
    # For larger p, the numbers p^2+i are larger, making smoothness harder
    # but p is larger too, so more factors are allowed
    # Let's try some larger primes
    if time.time() < deadline - 10:
        # Try primes in larger ranges
        for start in range(10003, 100000, 1000):
            if time.time() > deadline:
                break
            end = min(start + 1000, 100000)
            # Sieve a segment
            is_prime_seg = [True] * (end - start + 1)
            for i in range(2, int(math.isqrt(end)) + 1):
                if i >= start:
                    base_i = i
                else:
                    base_i = ((start + i - 1) // i) * i
                for j in range(base_i - start, end - start + 1, i):
                    if j >= 0:
                        is_prime_seg[j] = False
            for i in range(start, end):
                if time.time() > deadline:
                    break
                if i < 2 or not is_prime_seg[i - start]:
                    continue
                if not _is_prime(i):
                    continue
                k = 0
                base = i * i
                for offset in range(1, 201):
                    if time.time() > deadline:
                        break
                    n = base + offset
                    if not _is_p_smooth(n, i):
                        break
                    k = offset
                if k > best_k:
                    best_k = k
                    best_p = i
                    if best_k >= 200:
                        break
            if best_k >= 200:
                break
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_prime(n):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Deterministic Miller-Rabin for n < 3.3e24
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _sieve_primes(limit):
    """Sieve primes up to limit."""
    if limit < 2:
        return []
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return [i for i in range(2, limit + 1) if is_p[i]]

def _check_window_p_smooth(base, k_max, p, primes_le_p):
    """
    Check p^2+1 .. p^2+k_max for p-smoothness using segmented sieve.
    Returns the longest run length k such that p^2+1..p^2+k are all p-smooth.
    
    Strategy: For each number in [base+1, base+k_max], divide out all prime
    factors <= p. If remainder is 1, it's p-smooth. If remainder > 1, it has
    a prime factor > p (since we removed all factors <= p), so it's NOT p-smooth.
    
    We use a segmented approach: for each prime q <= p, we mark multiples of q
    in the window [base+1, base+k_max] and divide them out.
    """
    # rem[i] will hold the remaining part of (base + i) after dividing by all primes <= p
    # We only need to track up to k_max
    rem = [0] * (k_max + 1)
    for i in range(1, k_max + 1):
        rem[i] = base + i
    
    # For each prime q <= p, divide q out of all multiples in the window
    # We only need primes q where q <= p AND q <= max(rem[i]) which is base + k_max
    # But more importantly, we need q <= p (the smoothness bound)
    # Also, if q > base + k_max, q can't divide any number in the window
    max_val = base + k_max
    for q in primes_le_p:
        if q > p:
            break
        if q > max_val:
            break
        # Find first multiple of q in [base+1, base+k_max]
        first = ((base + 1 + q - 1) // q) * q
        if first < base + 1:
            first += q
        if first > base + k_max:
            continue
        for val in range(first, base + k_max + 1, q):
            idx = val - base
            while rem[idx] % q == 0:
                rem[idx] //= q
    # Now check: rem[i] == 1 means fully p-smooth
    # rem[i] > 1 means there's a prime factor > p remaining
    k = 0
    for i in range(1, k_max + 1):
        if rem[i] == 1:
            k = i
        else:
            break
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_k = 0
    best_p = 2
    
    # Precompute primes up to a reasonable limit for the smoothness check
    # We need primes up to p for the smoothness test, but p can be up to 10^9
    # For the segmented sieve, we only need primes up to min(p, sqrt(base+k_max))
    # Actually no: we need ALL primes <= p to divide out. But if q > sqrt(base+k_max),
    # then q can appear at most once as a factor. So we can handle large primes differently.
    # 
    # Key insight: For a number n in [p^2+1, p^2+200], any prime factor q of n
    # satisfies q <= n. If q > p, n is NOT p-smooth.
    # If q <= p, we need to divide it out.
    # 
    # For the segmented sieve approach:
    # - For small primes q (say q <= 10^6), we can directly sieve the window
    # - For large primes q (10^6 < q <= p), they can appear at most once in each n
    #   (since q^2 > 10^12 > p^2+200 for p <= 10^9... wait, p^2 can be up to 10^18)
    #   Actually for p up to 10^9, p^2 up to 10^18, so q^2 could be <= n.
    #   But the number of such large primes in the range [10^6, p] that actually
    #   divide any of our 200 numbers is very small.
    #
    # Simpler approach: Just use trial division with primes up to p, but optimized.
    # For each of the 200 numbers, divide by primes up to min(p, sqrt(n)).
    # After that, if remainder > 1, check if it's prime and <= p.
    #
    # Even simpler: precompute primes up to some limit (say 10^6), use them for
    # trial division. For the remaining part, use Miller-Rabin + check <= p.
    
    # Precompute primes up to 10^6 for trial division
    TRIAL_LIMIT = 10**6
    small_primes = _sieve_primes(TRIAL_LIMIT)
    
    # Strategy: iterate over candidate primes p, check the window
    # For efficiency, first do a quick check on p^2+1 using small primes
    # Then do full check only if p^2+1 passes
    
    # Generate candidate primes to test
    # Start with small primes, expand as needed
    
    # Phase 1: Check primes up to 200000
    SIEVE_LIMIT = 200000
    candidate_primes = _sieve_primes(SIEVE_LIMIT)
    
    for p in candidate_primes:
        if time.time() > deadline:
            break
        if p < 2:
            continue
        
        base = p * p
        k_max = 200
        
        # Quick pre-filter: check if p^2+1 is p-smooth using small primes only
        # If it fails, skip this p entirely
        n0 = base + 1
        temp = n0
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        if temp > 1:
            # temp has a prime factor > sqrt(n0) or is itself prime
            # Check if temp <= p (then it's fine) or if temp is prime and <= p
            if temp <= p:
                pass  # OK, p^2+1 is p-smooth
            elif _is_prime(temp) and temp <= p:
                pass  # OK
            else:
                # p^2+1 is NOT p-smooth, skip
                continue
        
        # Full check of the window using segmented sieve
        # We need primes up to min(p, sqrt(base + k_max)) for the sieve
        # But actually we need all primes <= p. For primes > sqrt(base+k_max),
        # they can divide each number at most once, so we handle them separately.
        
        # For the segmented sieve, use primes up to min(p, 10^6) which we have
        # For primes in (10^6, p], handle via direct division (at most once per number)
        
        # Use the small_primes list (up to 10^6) for the main sieve
        # Filter to only primes <= p
        primes_for_sieve = [q for q in small_primes if q <= p]
        
        k = _check_window_p_smooth(base, k_max, p, primes_for_sieve)
        
        # But wait: _check_window_p_smooth only divides by primes in primes_for_sieve
        # which are <= min(p, 10^6). For primes q with 10^6 < q <= p, we need to
        # also divide those out. Let me fix this.
        
        # Actually, let me reconsider. The function divides by all primes in
        # primes_for_sieve. If p > 10^6, there are primes between 10^6 and p
        # that we haven't divided out. Those primes, if they divide a number in
        # the window, would leave a remainder > 1, causing us to incorrectly
        # say the number is not p-smooth.
        #
        # Fix: After the segmented sieve with small primes, for each remaining
        # value > 1, check if it's prime and <= p. If so, it's p-smooth.
        # If it's composite, it has a factor > 10^6, and since all factors
        # <= 10^6 were removed, the remaining factor must be > 10^6.
        # If that factor is <= p, we need to verify.
        
        # Let me rewrite the check more carefully
        rem = [0] * (k_max + 1)
        for i in range(1, k_max + 1):
            rem[i] = base + i
        
        for q in primes_for_sieve:
            if q > base + k_max:
                break
            first = ((base + 1 + q - 1) // q) * q
            if first < base + 1:
                first += q
            if first > base + k_max:
                continue
            for val in range(first, base + k_max + 1, q):
                idx = val - base
                while rem[idx] % q == 0:
                    rem[idx] //= q
        
        # Now for each rem[i] > 1, check if it's p-smooth
        # rem[i] has no prime factors <= min(p, 10^6)
        # So any remaining prime factor is > 10^6
        # If rem[i] is prime and <= p, it's OK
        # If rem[i] is composite, it has at least two prime factors > 10^6
        #   so rem[i] > (10^6)^2 = 10^12
        #   For p <= 10^9, p^2 <= 10^18, so this is possible
        #   But if rem[i] is composite with all factors > 10^6,
        #   and we need all factors <= p, we need to check
        
        k = 0
        for i in range(1, k_max + 1):
            if rem[i] == 1:
                k = i
            elif rem[i] <= p:
                # Could be prime <= p, or composite with factors <= p
                # If prime <= p: OK
                # If composite: all factors must be <= p, but we already
                #   divided out all factors <= 10^6, so remaining factors
                #   are > 10^6. If rem[i] <= p and p <= 10^9, then
                #   rem[i] could be prime (OK) or product of two primes
                #   both > 10^6 and <= p. Need to check.
                if _is_prime(rem[i]):
                    k = i
                else:
                    # Composite, check if all factors <= p
                    # Since no factor <= 10^6 remains, smallest factor > 10^6
                    # rem[i] <= p <= 10^9, so at most 2 factors possible
                    # (10^6)^2 = 10^12 > 10^9, so at most 2 factors)
                    # Check if rem[i] has a factor <= p
                    # Try to find a factor
                    is_smooth = True
                    # rem[i] is composite, no factors <= 10^6
                    # Check if it's a product of two primes both <= p
                    # Since rem[i] <= 10^9, sqrt(rem[i]) <= ~31623
                    # But we said no factors <= 10^6, contradiction if rem[i] <= 10^12
                    # Wait: rem[i] has no factors <= 10^6, so if rem[i] is composite,
                    # its smallest prime factor > 10^6, so rem[i] > 10^12
                    # But rem[i] <= p <= 10^9 < 10^12, contradiction!
                    # So rem[i] must be prime if rem[i] <= p and rem[i] > 1
                    # and has no factors <= 10^6.
                    # This means the _is_prime check above should always return True
                    # for composite rem[i] <= p. Let's just trust the prime check.
                    is_smooth = False  # Should not happen
                    k = i if is_smooth else k
                    if not is_smooth:
                        break
            else:
                # rem[i] > p, definitely not p-smooth
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
    
    # Phase 2: Try larger primes if time remains
    if time.time() < deadline - 5 and best_k < 200:
        # Try primes in ranges beyond 200000
        # For larger p, the window p^2+1..p^2+200 has larger numbers
        # but p is also larger, so more factors are allowed
        for seg_start in range(200000, 1000000, 50000):
            if time.time() > deadline:
                break
            seg_end = min(seg_start + 50000, 1000000)
            # Segmented sieve for primes in [seg_start, seg_end]
            is_p_seg = [True] * (seg_end - seg_start + 1)
            for q in small_primes:
                if q * q > seg_end:
                    break
                first = ((seg_start + q - 1) // q) * q
                if first < q * q:
                    first = q * q
                for val in range(first, seg_end + 1, q):
                    is_p_seg[val - seg_start] = False
            for i in range(seg_start, seg_end + 1):
                if time.time() > deadline:
                    break
                if not is_p_seg[i - seg_start]:
                    continue
                if not _is_prime(i):
                    continue
                p = i
                base = p * p
                k_max = 200
                
                # Quick pre-filter on p^2+1
                n0 = base + 1
                temp = n0
                for q in small_primes:
                    if q > p:
                        break
                    if q * q > temp:
                        break
                    while temp % q == 0:
                        temp //= q
                if temp > 1:
                    if temp <= p:
                        pass
                    elif _is_prime(temp) and temp <= p:
                        pass
                    else:
                        continue
                
                # Full window check
                primes_for_sieve = [q for q in small_primes if q <= p]
                rem = [0] * (k_max + 1)
                for i in range(1, k_max + 1):
                    rem[i] = base + i
                for q in primes_for_sieve:
                    if q > base + k_max:
                        break
                    first = ((base + 1 + q - 1) // q) * q
                    if first < base + 1:
                        first += q
                    if first > base + k_max:
                        continue
                    for val in range(first, base + k_max + 1, q):
                        idx = val - base
                        while rem[idx] % q == 0:
                            rem[idx] //= q
                
                k = 0
                for i in range(1, k_max + 1):
                    if rem[i] == 1:
                        k = i
                    elif rem[i] <= p and _is_prime(rem[i]):
                        k = i
                    else:
                        break
                
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break
            if best_k >= 200:
                break
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize there's a flaw in my logic. Let me reconsider.

When `rem[i] <= p` and `rem[i] > 1`, and we've already divided out all primes ≤ min(p, 10^6):
- If p ≤ 10^6: all primes ≤ p were divided out, so `rem[i]` has no prime factors ≤ p. If `rem[i] > 1`, it must have a prime factor > p. So `rem[i] > 1` means NOT p-smooth. My code incorrectly accepts `rem[i] <= p` as smooth.
- If p > 10^6: primes between 10^6 and p were NOT divided out. So `rem[i]` could have prime factors in (10^6, p]. If `rem[i]` is prime and ≤ p, it's OK. If `rem[i]` is composite, its smallest factor > 10^6, so `rem[i] > 10^12`. But `rem[i] ≤ p ≤ 10^9 < 10^12`, contradiction. So `rem[i]` must be prime.

So the fix is: if `rem[i] > 1` and `rem[i] <= p`, check if it's prime. If prime, OK. If not prime (shouldn't happen for p ≤ 10^9), not OK.

But the bigger issue: for p ≤ 10^6, `rem[i] > 1` means NOT p-smooth (all factors ≤ p were removed). My current code would incorrectly accept it if `rem[i] <= p`. Let me fix this properly.

