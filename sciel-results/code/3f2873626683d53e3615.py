"""Rewritten program: random pairwise-sum frequency sampling to find N with many powerful representations."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from collections import Counter

_START_TIME = time.time()
DEADLINE = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _is_powerful(n):
    """Check if n is powerful: every prime factor appears with exponent >= 2."""
    if n == 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            if exp == 1:
                return False
        d += 1 if d == 2 else 2
    # temp is now the remaining prime factor (if any)
    if temp > 1:
        return False  # remaining factor has exponent 1
    return True

def _gen_powerful_up_to(limit):
    """Generate all powerful numbers <= limit using squarefree * square^2 form."""
    # Powerful number: product of p_i^{e_i} where e_i >= 2.
    # Equivalently: n = s^2 * q where q is squarefree and for each prime p|q, p^2 | n.
    # Better: n = product p_i^{e_i}, e_i >= 2.
    # Enumerate: for squarefree part sf and square part sq, n = sf * sq^2 is powerful iff
    # for every prime p dividing sf, p^2 divides n, i.e., p divides sq.
    # Simplest: enumerate directly by prime powers.
    # A powerful number <= 10^12: product of p^e with e>=2.
    # Max prime: p^2 <= 10^12 => p <= 10^6.
    # For each powerful number, it's a product of prime powers p_i^{e_i}, e_i>=2.
    
    powerful = set([1])
    
    # Generate all prime powers p^e <= limit with e >= 2
    # Primes up to sqrt(limit) = 10^6
    # But we need to combine them multiplicatively.
    
    # Alternative: n is powerful iff n = a^2 * b where b is squarefree and
    # rad(b) | a. Let's enumerate by the "square part" and "squarefree part".
    
    # Direct enumeration: powerful numbers are products of p^e (e>=2).
    # Let's generate primes up to limit^(1/2) = 10^6 for the largest prime factor.
    # Actually for limit=10^12, max prime p with p^2<=limit is 10^6.
    
    # Use a sieve to get primes up to 10^6
    max_prime = isqrt(limit)
    if max_prime < 2:
        return [1]
    
    # Sieve of Eratosthenes up to max_prime
    sieve = bytearray(b'\x01') * (max_prime + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(max_prime) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, max_prime + 1) if sieve[i]]
    
    # Generate powerful numbers by DFS over primes
    # n = product of p_i^{e_i} where e_i >= 2, and n <= limit
    # We process primes in order; for each prime, we can use it with exponent 2,3,4,... or skip it.
    # But we need to be careful: once we skip a prime, we can't come back.
    
    # Actually, a powerful number has a specific set of prime factors, each with exponent >= 2.
    # So we enumerate subsets of primes with assigned exponents.
    
    # More efficient: recursive generation
    result = [1]
    
    def dfs(idx, current):
        if current > limit:
            return
        if idx >= len(primes):
            if current > 1:
                result.append(current)
            return
        p = primes[idx]
        # Option 1: skip this prime
        dfs(idx + 1, current)
        # Option 2: include this prime with exponent e >= 2
        power = p * p
        while power <= limit:
            new_val = current * power
            if new_val > limit:
                break
            result.append(new_val)
            # Continue with higher exponents or next primes
            dfs(idx + 1, new_val)
            power *= p
    
    dfs(0, 1)
    
    return sorted(set(result))

def _count_pairs(N, powerful_list, powerful_set):
    """Count and return pairs (a,b) with a<=b, a+b=N, both powerful."""
    pairs = []
    for a in powerful_list:
        if a > N // 2:
            break
        b = N - a
        if b < a:
            continue
        if b in powerful_set:
            pairs.append([a, b])
    return pairs

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Phase 1: Generate powerful numbers up to 10^12
    # But generating all powerful numbers up to 10^12 might be large.
    # Let's estimate: number of powerful numbers up to 10^12 is about O(sqrt(n) * log factors) ~ a few million.
    # Actually the count of powerful numbers up to x is ~ C * sqrt(x) for some constant.
    # For x=10^12, sqrt(x)=10^6, so maybe ~10^6 to 10^7 powerful numbers.
    
    # Let's generate up to 10^12 but be mindful of time.
    limit = 10**12
    
    # Generate powerful numbers
    powerful_list = _gen_powerful_up_to(limit)
    powerful_set = set(powerful_list)
    
    if _elapsed() > DEADLINE - 30:
        # Fallback: just pick a small N
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: Random pairwise-sum sampling to identify candidate N values
    # Sample many random pairs (a, b) from powerful_list and record their sums.
    n_samples = 500000
    sum_counter = Counter()
    
    # To ensure a <= b and a + b <= 2*limit, we sample from the list.
    # Use reservoir or random sampling.
    n_pow = len(powerful_list)
    
    # For efficiency, sample indices
    rng = random.Random(42)
    
    for _ in range(n_samples):
        i = rng.randint(0, n_pow - 1)
        j = rng.randint(0, n_pow - 1)
        a = powerful_list[i]
        b = powerful_list[j]
        if a > b:
            a, b = b, a
        s = a + b
        if s <= 2 * limit:  # N can be up to 2*limit since a,b <= limit
            sum_counter[s] += 1
    
    if _elapsed() > DEADLINE - 30:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 3: Get top candidates and verify
    top_candidates = [s for s, _ in sum_counter.most_common(50)]
    
    best_N = None
    best_pairs = []
    
    for N in top_candidates:
        if _elapsed() > DEADLINE - 10:
            break
        if N < 2 or N > 2 * limit:
            continue
        pairs = _count_pairs(N, powerful_list, powerful_set)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
    
    # Also try some structured candidates: multiples of small powerful numbers
    if _elapsed() < DEADLINE - 20:
        # Try N = k * m where m is a small powerful number and k is moderate
        # Actually, let's try N values that are sums of two small powerful numbers with high multiplicity
        # Try: N = a + b where a and b are from the smallest powerful numbers
        small_pow = powerful_list[:1000]
        small_set = set(small_pow)
        
        # Count representations using only small powerful numbers first (fast)
        small_counter = Counter()
        for a in small_pow:
            for b in small_pow:
                if a <= b:
                    small_counter[a + b] += 1
        
        top_small = [s for s, _ in small_counter.most_common(20)]
        for N in top_small:
            if _elapsed() > DEADLINE - 10:
                break
            if N < 2 or N > 2 * limit:
                continue
            pairs = _count_pairs(N, powerful_list, powerful_set)
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_N = N
    
    # Ensure we have at least one valid result
    if best_N is None or not best_pairs:
        # Find any valid N
        for N in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            pairs = _count_pairs(N, powerful_list, powerful_set)
            if pairs:
                best_N = N
                best_pairs = pairs
                break
    
    if best_N is None:
        best_N = 2
        best_pairs = [[1, 1]]
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
