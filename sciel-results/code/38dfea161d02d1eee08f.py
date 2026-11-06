import math
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    LIMIT = 10**18
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors
    
    def tau_perp(n, factors=None):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if factors is None:
            factors = []
            temp = n
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
                p += 1 if p == 2 else 2
            if temp > 1:
                factors.append((temp, 1))
        
        divisors = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score, len(divisors)
    
    best_n = 2
    best_score = 0
    
    # First, establish a strong baseline: product of first 15 primes
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_base = 1
    for p in primes_15:
        n_base *= p
    
    if n_base <= LIMIT:
        score_base, _ = tau_perp(n_base, [(p, 1) for p in primes_15])
        best_n = n_base
        best_score = score_base
    
    # Strategy: try to maximize the number of coprime adjacent pairs.
    # Key insight: For a squarefree number with k primes, there are 2^k divisors.
    # Two divisors are coprime if they share no prime factors.
    # In the sorted list of divisors, we want adjacent pairs to be coprime.
    # 
    # For squarefree numbers, the divisors can be viewed as subsets of primes.
    # The sorted order is by numerical value.
    # 
    # Let's try various configurations:
    # 1. Squarefree with many primes (up to 15)
    # 2. Numbers with some exponents > 1 but still many distinct primes
    # 3. Highly composite-like numbers
    
    # Try 14 primes with one exponent 2
    for i in range(15):
        if time.time() - start_time > DEADLINE:
            return best_n
        
        primes_test = primes_15[:14]
        exps = [1] * 14
        exps[i] = 2  # Give one prime exponent 2
        
        n_test = 1
        for p, e in zip(primes_test, exps):
            n_test *= p ** e
        
        if n_test <= LIMIT:
            factors = list(zip(primes_test, exps))
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    # Try 13 primes with two exponents 2
    for i in range(13):
        for j in range(i + 1, 13):
            if time.time() - start_time > DEADLINE:
                return best_n
            
            primes_test = primes_15[:13]
            exps = [1] * 13
            exps[i] = 2
            exps[j] = 2
            
            n_test = 1
            for p, e in zip(primes_test, exps):
                n_test *= p ** e
            
            if n_test <= LIMIT:
                factors = list(zip(primes_test, exps))
                score, _ = tau_perp(n_test, factors)
                if score > best_score:
                    best_n = n_test
                    best_score = score
    
    # Try some highly composite-like structures
    # These have many divisors but fewer distinct primes
    hc_candidates = [
        # (exponents for primes 2,3,5,7,11,13,17,19,23,...)
        ([60, 36, 24, 18, 12, 9, 6, 4, 3, 2, 2, 2, 2, 2], None),
        ([54, 32, 22, 16, 11, 8, 5, 4, 3, 2, 2, 2], None),
        ([48, 30, 20, 14, 10, 7, 5, 4, 3, 2, 2], None),
        ([42, 26, 18, 12, 9, 6, 5, 4, 3, 2], None),
        ([36, 22, 16, 11, 8, 6, 5, 4, 3], None),
        ([30, 18, 14, 10, 7, 6, 5, 4], None),
        ([24, 16, 12, 9, 7, 5, 4, 3], None),
        ([18, 14, 10, 8, 6, 5, 4], None),
        ([12, 10, 8, 6, 5, 4, 3], None),
        ([8, 6, 5, 4, 3, 2, 2], None),
        ([6, 5, 4, 3, 2, 2, 2], None),
        ([4, 3, 3, 2, 2, 2, 2], None),
        ([3, 3, 2, 2, 2, 2, 2, 2], None),
        ([2, 2, 2, 2, 2, 2, 2, 2, 2], None),
        ([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], None),
        ([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], None),
    ]
    
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for exps, _ in hc_candidates:
        if time.time() - start_time > DEADLINE:
            return best_n
        
        n_test = 1
        for i, e in enumerate(exps):
            if i < len(base_primes):
                n_test *= base_primes[i] ** e
            else:
                break
        
        if n_test <= LIMIT and n_test >= 2:
            factors = [(base_primes[i], exps[i]) for i in range(len(exps)) if i < len(base_primes)]
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    # Try mixed structures: some primes with exponent 1, some with higher
    # Focus on keeping many distinct primes while adding some exponents
    mixed_patterns = [
        # (primes, exponents)
        ([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43], [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], [4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37], [5, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1]),
        ([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], [6, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1]),
        ([2, 3, 5, 7, 11, 13, 17, 19, 23, 29], [8, 6, 5, 4, 3, 2, 2, 1, 1, 1]),
        ([2, 3, 5, 7, 11, 13, 17, 19, 23], [10, 8, 6, 5, 4, 3, 2, 2, 1]),
        ([2, 3, 5, 7, 11, 13, 17, 19], [12, 10, 8, 6, 5, 4, 3, 2]),
        ([2, 3, 5, 7, 11, 13, 17], [15, 12, 10, 8, 6, 5, 4]),
        ([2, 3, 5, 7, 11, 13], [18, 15, 12, 10, 8, 6]),
        ([2, 3, 5, 7, 11], [22, 18, 15, 12, 10]),
        ([2, 3, 5, 7], [27, 22, 18, 15]),
        ([2, 3, 5], [33, 27, 22]),
    ]
    
    for primes_test, exps in mixed_patterns:
        if time.time() - start_time > DEADLINE:
            return best_n
        
        n_test = 1
        for p, e in zip(primes_test, exps):
            n_test *= p ** e
        
        if n_test <= LIMIT and n_test >= 2:
            factors = list(zip(primes_test, exps))
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    # Try a systematic search over exponent vectors for the first k primes
    # where exponents are non-increasing (since we want to maximize divisor count
    # for a given number of primes, smaller primes should have larger exponents)
    
    # For k=8 primes (2,3,5,7,11,13,17,19), try various exponent patterns
    primes_8 = [2, 3, 5, 7, 11, 13, 17, 19]
    
    # Generate non-increasing exponent sequences
    def gen_non_increasing_exps(k, max_exp):
        """Generate non-increasing sequences of length k with values 1..max_exp."""
        if k == 0:
            yield ()
            return
        for first in range(max_exp, 0, -1):
            for rest in gen_non_increasing_exps(k - 1, first):
                yield (first,) + rest
    
    # Try for 8 primes with max exponent 8
    for exps in gen_non_increasing_exps(8, 8):
        if time.time() - start_time > DEADLINE:
            return best_n
        
        n_test = 1
        for p, e in zip(primes_8, exps):
            n_test *= p ** e
            if n_test > LIMIT:
                break
        
        if n_test <= LIMIT and n_test >= 2:
            factors = list(zip(primes_8, exps))
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    # Try for 9 primes with max exponent 6
    primes_9 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    for exps in gen_non_increasing_exps(9, 6):
        if time.time() - start_time > DEADLINE:
            return best_n
        
        n_test = 1
        for p, e in zip(primes_9, exps):
            n_test *= p ** e
            if n_test > LIMIT:
                break
        
        if n_test <= LIMIT and n_test >= 2:
            factors = list(zip(primes_9, exps))
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    # Try for 10 primes with max exponent 4
    primes_10 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    for exps in gen_non_increasing_exps(10, 4):
        if time.time() - start_time > DEADLINE:
            return best_n
        
        n_test = 1
        for p, e in zip(primes_10, exps):
            n_test *= p ** e
            if n_test > LIMIT:
                break
        
        if n_test <= LIMIT and n_test >= 2:
            factors = list(zip(primes_10, exps))
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    # Try for 11 primes with max exponent 3
    primes_11 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    for exps in gen_non_increasing_exps(11, 3):
        if time.time() - start_time > DEADLINE:
            return best_n
        
        n_test = 1
        for p, e in zip(primes_11, exps):
            n_test *= p ** e
            if n_test > LIMIT:
                break
        
        if n_test <= LIMIT and n_test >= 2:
            factors = list(zip(primes_11, exps))
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    # Try for 12 primes with max exponent 2
    primes_12 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for exps in gen_non_increasing_exps(12, 2):
        if time.time() - start_time > DEADLINE:
            return best_n
        
        n_test = 1
        for p, e in zip(primes_12, exps):
            n_test *= p ** e
            if n_test > LIMIT:
                break
        
        if n_test <= LIMIT and n_test >= 2:
            factors = list(zip(primes_12, exps))
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_n = n_test
                best_score = score
    
    return best_n
