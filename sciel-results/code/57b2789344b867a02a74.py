import math
import time
import random
from array import array

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
        # Deterministic Miller-Rabin for 64-bit
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Witnesses sufficient for n < 3.4 * 10^14
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if 1 < g < n:
                return g

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        if is_prime(n):
            return n
        factors = []
        while n > 1:
            if is_prime(n):
                factors.append(n)
                break
            d = pollard_brent(n)
            if d == n:
                # Retry with different parameters
                d = pollard_brent(n)
                if d == n:
                    # Fallback: trial division up to sqrt
                    limit = int(math.isqrt(n)) + 1
                    d = None
                    for i in range(2, limit):
                        if n % i == 0:
                            d = i
                            break
                    if d is None:
                        factors.append(n)
                        break
            factors.append(d)
            n //= d
        return max(factors)

    # Pre-compute small primes up to 100000 for trial division
    limit = 100000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Segmented sieve to find primes in [low, high]
    def primes_in_range(low, high):
        if high < 2:
            return []
        if low < 2:
            low = 2
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        for p in small_primes:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = 0
        result = []
        for i in range(size):
            if is_prime_seg[i]:
                result.append(low + i)
        return result

    def is_p_smooth(n, p):
        """Check if n has all prime factors <= p"""
        if n <= 1:
            return True
        if p < 2:
            return n <= 1
        
        # Trial divide by small primes up to min(p, 100000)
        rem = n
        for sp in small_primes:
            if sp > p:
                break
            if rem % sp == 0:
                while rem % sp == 0:
                    rem //= sp
                if rem == 1:
                    return True
        
        if rem == 1:
            return True
        
        # rem is now composed of primes > 100000 (or > p if p < 100000)
        # If rem is prime, check if rem <= p
        if is_prime(rem):
            return rem <= p
        
        # rem is composite. Find its largest prime factor.
        # If largest prime factor <= p, then smooth.
        return largest_prime_factor(rem) <= p

    best_k = 0
    best_p = 2
    
    # Strategy: Search for primes with long runs of p-smooth numbers after p^2
    # Key insight: For large p, p^2+1 through p^2+k are all ~p^2 in magnitude
    # For them to be p-smooth, they need to have no prime factor > p
    # This is more likely for smaller p where the "smoothness" requirement is easier
    
    # Phase 1: Search smaller primes [10^4, 10^6] - higher density, easier smoothness
    # Phase 2: Search medium primes [10^6, 10^8]
    # Phase 3: Search larger primes [10^8, 10^9]
    
    # Use deterministic sweep with random restarts within each phase
    phase_boundaries = [(10**4, 10**6, 40.0), (10**6, 10**8, 75.0), (10**8, 10**9, DEADLINE)]
    
    for seg_low_bound, seg_high_bound, phase_deadline in phase_boundaries:
        while time.time() < phase_deadline:
            # Pick a random segment
            seg_size = min(2000, seg_high_bound - seg_low_bound)
            if seg_size <= 0:
                break
            seg_low = random.randint(seg_low_bound, max(seg_low_bound, seg_high_bound - seg_size))
            seg_high = min(seg_low + seg_size, seg_high_bound)
            
            primes = primes_in_range(seg_low, seg_high)
            
            for p in primes:
                if time.time() >= phase_deadline:
                    break
                
                p2 = p * p
                k = 0
                while k < 200:
                    val = p2 + k + 1
                    if not is_p_smooth(val, p):
                        break
                    k += 1
                
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
