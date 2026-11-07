import math
import time
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin
    
    # Phase 1: Linear sieve for small k
    N = 8000000
    
    # Use array module for memory efficiency
    sigma = array('q', [0]) * (N + 1)
    p_pow = array('q', [0]) * (N + 1)
    sigma_rest = array('q', [0]) * (N + 1)
    spf = array('i', [0]) * (N + 1)
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_rest[1] = 1
    
    primes = []
    primes.append(2)
    spf[2] = 2
    sigma[2] = 3
    p_pow[2] = 2
    sigma_rest[2] = 1
    
    for i in range(3, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            p_pow[i] = i
            sigma_rest[i] = 1
            
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                p_pow[i * p] = p_pow[i] * p
                sigma_rest[i * p] = sigma_rest[i]
                sigma[i * p] = sigma[i] + p_pow[i] * p * sigma_rest[i]
                break
            else:
                p_pow[i * p] = p
                sigma_rest[i * p] = sigma[i]
                sigma[i * p] = sigma[i] * (1 + p)
                
        if time.time() > deadline - 40.0:
            break

    # Find the actual max k processed
    max_k_processed = N
    while max_k_processed > 1 and sigma[max_k_processed] == 0:
        max_k_processed -= 1
    
    # Group by k*sigma(k)
    groups = {}
    for k in range(1, max_k_processed + 1):
        c = k * sigma[k]
        if c not in groups:
            groups[c] = []
        groups[c].append(k)
        if time.time() > deadline - 30.0:
            break

    # Find best group
    best_group = []
    best_c = 0
    for c, xs in groups.items():
        if len(xs) > len(best_group):
            best_group = xs
            best_c = c
            
    best_count = len(best_group)
    best_M = int(best_c)
    
    # Phase 2: Targeted search for larger k values
    # Focus on k that are highly composite (products of small primes)
    # Compute sigma(k) for specific k values in larger range
    def sigma_individual(k):
        """Compute sigma(k) by factorization"""
        if k <= max_k_processed and sigma[k] > 0:
            return sigma[k]
        n = k
        result = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                power = 0
                while n % d == 0:
                    n //= d
                    power += 1
                result *= (d**(power+1) - 1) // (d - 1)
            d += 1 if d == 2 else 2
        if n > 1:
            result *= (n*n + n)
        return result
    
    # Search for k values in range [max_k_processed+1, 20_000_000]
    # that are products of small primes (highly likely to have collisions)
    if time.time() < deadline - 20.0:
        # Generate candidate k values: products of small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        candidates = set()
        
        # Generate numbers that are 7-smooth or similar
        # k = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g
        # with k in [max_k_processed, 20_000_000]
        
        def gen_smooth(lo, hi, primes_list, max_depth=7):
            if max_depth == 0:
                if lo <= 1 <= hi:
                    yield 1
                return
            p = primes_list[0]
            rest = primes_list[1:]
            power = 1
            while power <= hi:
                for sub in gen_smooth(max(lo // power, 1), hi // power, rest, max_depth - 1):
                    val = power * sub
                    if lo <= val <= hi:
                        yield val
                power *= p
                if power > hi:
                    break
        
        # Generate 17-smooth numbers in range
        lo = max_k_processed + 1
        hi = min(20000000, int(10**9))  # cap at 20M for now
        
        count_checked = 0
        for k in gen_smooth(lo, hi, small_primes, 7):
            if time.time() > deadline - 15.0:
                break
            s = sigma_individual(k)
            c = k * s
            if c not in groups:
                groups[c] = []
            groups[c].append(k)
            count_checked += 1
            
            if count_checked % 100000 == 0 and time.time() > deadline - 15.0:
                break
        
        # Re-find best group
        for c, xs in groups.items():
            if len(xs) > len(best_group):
                best_group = xs
                best_c = c
                best_count = len(xs)
                best_M = int(c)
    
    # Verify and return
    if best_count >= 3:
        sqrt_M = math.isqrt(best_M)
        valid_ks = [int(x) for x in best_group if x <= sqrt_M]
        if len(valid_ks) >= 3:
            return {'M': best_M, 'k': valid_ks}
    
    # Fallback
    return {'M': 333312, 'k': [336, 372, 434]}
