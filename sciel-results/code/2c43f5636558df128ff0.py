import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Precompute primes up to 2*10^6 for better coverage of small factors
    P_MAX = 2000000
    is_prime_sieve = bytearray(b'\x01') * (P_MAX + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(P_MAX)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, P_MAX + 1, i):
                is_prime_sieve[j] = 0
    primes = [i for i in range(2, P_MAX + 1) if is_prime_sieve[i]]
    # Separate small primes for fast trial division
    small_primes = [q for q in primes if q <= 10000]

    def is_p_smooth_fast(n, p, small_primes_list, primes_list):
        """Check if n is p-smooth. Returns True if all prime factors <= p."""
        if n <= 1:
            return True
        rem = n
        # Trial divide by small primes first
        for q in small_primes_list:
            if q > p:
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
            if rem == 1:
                return True
        if rem == 1:
            return True
        # If remaining factor is <= p, we're done
        if rem <= p:
            return True
        # Need to check if rem is prime and <= p, or composite with all factors <= p
        # Since rem > p, we need to verify rem has no prime factor > p
        # But rem could be prime > p (fail) or composite with factors <= p (pass)
        # Check if rem is prime
        if _is_prime_deterministic(rem):
            return rem <= p
        # rem is composite; try to find a factor
        # If rem has a factor > p, it fails
        # Try dividing by primes up to min(p, sqrt(rem))
        for q in primes_list:
            if q > p:
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
            if rem == 1:
                return True
        if rem == 1:
            return True
        if rem <= p:
            return True
        # rem > p and we couldn't fully factor it with primes <= p
        # Check if rem itself is prime
        if _is_prime_deterministic(rem):
            return False  # rem is prime > p
        # rem is composite but we couldn't factor it completely
        # This shouldn't happen if our prime list is sufficient
        # Use a more aggressive check
        return _check_composite_p_smooth(rem, p, primes_list)

    def _is_prime_deterministic(n):
        """Deterministic Miller-Rabin for n < 3.3*10^24"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Bases sufficient for n < 3.3 * 10^24
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _check_composite_p_smooth(rem, p, primes_list):
        """Additional check for composite rem > p: verify all factors <= p"""
        if rem <= p:
            return True
        if _is_prime_deterministic(rem):
            return False
        # Try to factor rem using primes up to p
        for q in primes_list:
            if q > p:
                break
            if q * q > rem:
                break
            while rem % q == 0:
                rem //= q
            if rem == 1:
                return True
        if rem == 1:
            return True
        if rem <= p:
            return True
        if _is_prime_deterministic(rem):
            return False
        return False

    def find_max_k(p, max_k=200):
        """Find the maximum k such that p^2+1..p^2+k are all p-smooth."""
        k = 0
        p2 = p * p
        for j in range(1, max_k + 1):
            if not is_p_smooth_fast(p2 + j, p, small_primes, primes):
                break
            k += 1
        return k

    best_k = 1
    best_p = 2

    # Phase 1: Quick screen - check p^2+1 for primes in a targeted range
    # Larger primes are more likely to have longer runs
    # Focus on primes between 100000 and P_MAX
    candidates = []
    # Check primes in descending order (larger primes first)
    for p in reversed(primes):
        if p < 100000:
            break
        if time.time() - start_time > DEADLINE - 60:
            break
        p2 = p * p
        if is_p_smooth_fast(p2 + 1, p, small_primes, primes):
            candidates.append(p)
    
    # Phase 2: Full check on candidates, sorted by potential (larger first)
    candidates.sort(reverse=True)
    for p in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 3: If time remains, check medium primes
    if time.time() - start_time < DEADLINE - 20:
        for p in reversed(primes):
            if p >= 100000:
                continue
            if p < 50000:
                break
            if time.time() - start_time > DEADLINE - 10:
                break
            p2 = p * p
            if not is_p_smooth_fast(p2 + 1, p, small_primes, primes):
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 4: Random sampling of larger primes (up to 10^9) using probabilistic primality
    if time.time() - start_time < DEADLINE - 30:
        # Try some random primes in [10^6, 10^9]
        rng = random.Random(42)
        for _ in range(500):
            if time.time() - start_time > DEADLINE - 10:
                break
            p = rng.randint(1000000, 1000000000)
            # Find next prime >= p
            while not _is_prime_deterministic(p):
                p += 2
                if p > 1000000000:
                    break
            if p > 1000000000:
                continue
            p2 = p * p
            if is_p_smooth_fast(p2 + 1, p, small_primes, primes):
                k = find_max_k(p)
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
