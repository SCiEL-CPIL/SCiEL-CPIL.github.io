"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
from math import isqrt

# EVOLVE-BLOCK-START
def _sigma_small(n, sigma_arr):
    """Compute sigma(n) using precomputed array for small n, trial division for large."""
    if n < len(sigma_arr):
        return sigma_arr[n]
    # Trial division
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            p_pow = 1
            p_sum = 1
            while temp % d == 0:
                temp //= d
                p_pow *= d
                p_sum += p_pow
            result *= p_sum
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        p_pow = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_pow)
            p_pow *= p
        divs = new_divs
    return divs

def construct():
    # Precompute sigma for small numbers via sieve
    limit = 2000000
    sigma_arr = [0] * (limit + 1)
    for i in range(1, limit + 1):
        for j in range(i, limit + 1, i):
            sigma_arr[j] += i
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Search strategy: try M values that are products of small prime powers
    # Focus on M in ranges where we might find many solutions
    # Try M = 2^a * 3^b * 5^c * 7^d * ... for various combinations
    
    import time
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    # First, try known good M values and their multiples
    # Then do a broader search
    
    # Candidate M values to try: highly composite-like numbers
    # Generate candidates as products of small primes raised to powers
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    def generate_candidates(max_val=10**18):
        """Generate candidate M values that are products of small prime powers."""
        candidates = set()
        
        # Start with base values
        base = 1
        for p in primes:
            val = 1
            while val <= max_val:
                base *= val
                if base <= max_val:
                    candidates.add(base)
                val *= p
            base //= val
        
        # Also add multiples of known good M
        known_good = [333312, 720720, 1081080, 2162160, 4324320, 8648640]
        for kg in known_good:
            for mult in range(1, 1000):
                m = kg * mult
                if m <= max_val:
                    candidates.add(m)
        
        return candidates
    
    # Generate a set of candidate M values to check
    # Focus on M that are "smooth" (small prime factors) as they have many divisors
    
    candidates = set()
    
    # Generate M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f for small exponents
    for a in range(0, 61):  # 2^60 > 10^18
        v2 = 1 << a
        if v2 > 10**18:
            break
        for b in range(0, 40):
            v3 = v2 * (3 ** b)
            if v3 > 10**18:
                break
            for c in range(0, 26):
                v5 = v3 * (5 ** c)
                if v5 > 10**18:
                    break
                for d in range(0, 22):
                    v7 = v5 * (7 ** d)
                    if v7 > 10**18:
                        break
                    for e in range(0, 18):
                        v11 = v7 * (11 ** e)
                        if v11 > 10**18:
                            break
                        for f in range(0, 16):
                            v13 = v11 * (13 ** f)
                            if v13 > 10**18:
                                break
                            candidates.add(v13)
                    if time.time() - start_time > time_limit:
                        break
                if time.time() - start_time > time_limit:
                    break
            if time.time() - start_time > time_limit:
                break
        if time.time() - start_time > time_limit:
            break
    
    # Sort candidates by number of divisors (more divisors = more potential k)
    # But computing divisor count for all is expensive, so just sort by value and check
    # Actually, let's be smarter: check candidates in order of likely richness
    
    # Convert to list and sort by a heuristic: number of prime factors with multiplicity
    cand_list = list(candidates)
    
    # Sort by a heuristic: prefer M with many small prime factors
    def divisor_count_heuristic(n):
        """Rough estimate of number of divisors."""
        temp = n
        count = 1
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                count *= (exp + 1)
            d += 1
        if temp > 1:
            count *= 2
        return count
    
    # Only sort a subset to save time
    # Check the most promising candidates first
    # Use a simple heuristic: M with many small prime factors
    
    # For efficiency, limit the number of M values we check
    max_checks = 5000
    
    # Sort by a quick heuristic
    def quick_heuristic(n):
        """Quick check: count small prime factors."""
        temp = n
        count = 0
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while temp % p == 0:
                temp //= p
                count += 1
        return count
    
    # Pick top candidates by heuristic
    # But this is O(n log n) for large n, so sample if needed
    if len(cand_list) > 100000:
        # Sample
        import random
        random.seed(42)
        cand_list = random.sample(cand_list, 100000)
    
    # Sort by heuristic (more small prime factors = more divisors)
    cand_list.sort(key=quick_heuristic, reverse=True)
    
    for M in cand_list[:max_checks]:
        if time.time() - start_time > time_limit:
            break
        
        if M < 2:
            continue
        
        # Factorize M
        factors = _factorize(M)
        if not factors:
            continue
        
        # Generate all divisors
        divs = _divisors_from_factors(factors)
        
        # Find k <= sqrt(M) with k*sigma(k) = M
        sqrt_M = isqrt(M)
        solutions = []
        
        for k in divs:
            if k > sqrt_M:
                continue
            # Check if k * sigma(k) == M
            sig_k = _sigma_small(k, sigma_arr)
            if k * sig_k == M:
                solutions.append(k)
        
        if len(solutions) > best_count:
            best_count = len(solutions)
            best_M = M
            best_k = sorted(solutions)
    
    # Also try some specific constructions
    # M = product of primes where we can construct k values
    
    # Check a few more specific M values that are known to have many solutions
    extra_Ms = [333312, 720720, 1081080, 2162160, 4324320, 8648640, 17297280, 34594560]
    for M in extra_Ms:
        if time.time() - start_time > time_limit:
            break
        if M > 10**18:
            continue
        factors = _factorize(M)
        if not factors:
            continue
        divs = _divisors_from_factors(factors)
        sqrt_M = isqrt(M)
        solutions = []
        for k in divs:
            if k > sqrt_M:
                continue
            sig_k = _sigma_small(k, sigma_arr)
            if k * sig_k == M:
                solutions.append(k)
        if len(solutions) > best_count:
            best_count = len(solutions)
            best_M = M
            best_k = sorted(solutions)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(x) for x in best_k]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
