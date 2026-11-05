"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import heapq

# EVOLVE-BLOCK-START

def _largest_prime_factor(n):
    """Return the largest prime factor of n. P(1)=1."""
    if n <= 1:
        return 1
    result = 1
    # factor out 2
    while n % 2 == 0:
        result = 2
        n //= 2
    # factor out odd numbers up to sqrt(n)
    f = 3
    while f * f <= n:
        while n % f == 0:
            result = f
            n //= f
        f += 2
    if n > 1:
        result = n
    return result

def construct():
    target = 123457
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Precompute smallest prime factor (SPF) for all numbers up to target.
    max_n = target
    spf = list(range(max_n + 1))  # smallest prime factor
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_pf(n):
        if n <= 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            result = p
            while n % p == 0:
                n //= p
        return result
    
    # Strategy 1: K-smooth search. For increasing K, generate all K-smooth numbers
    # up to target and check if target - smooth is also K-smooth.
    # This directly finds the minimal K = max(P(a), P(b)).
    
    # Get all primes up to target
    primes = []
    for i in range(2, max_n + 1):
        if spf[i] == i:
            primes.append(i)
    
    best_a = 1
    best_b = target - 1
    best_score = max(largest_pf(best_a), largest_pf(best_b))
    
    # Incrementally build K-smooth set
    smooth_set = {1}
    
    for K in primes:
        if time.time() > deadline:
            break
        
        # Add all multiples of K to the smooth set (numbers whose prime factors are all <= K)
        new_smooth = set()
        for s in list(smooth_set):
            # Multiply by powers of K
            val = s
            while val * K <= max_n:
                val *= K
                new_smooth.add(val)
        
        smooth_set.update(new_smooth)
        
        # Check if any smooth number a has target - a also in smooth_set
        for a in smooth_set:
            if a < 1 or a > max_n - 1:
                continue
            b = target - a
            if 1 <= b <= max_n and b in smooth_set:
                # Found a valid pair with max(P(a), P(b)) <= K
                # Since we're iterating K in order, this is optimal
                best_a = a
                best_b = b
                best_score = K
                return [best_a, best_b]
    
    # Strategy 2: Fallback to brute force if K-smooth didn't find a solution
    # (This shouldn't happen since K=target is always valid with a=1, b=target-1)
    half = target // 2
    for a in range(1, half + 1):
        b = target - a
        pa = largest_pf(a)
        pb = largest_pf(b)
        score = pa if pa > pb else pb
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            if best_score == 1:
                break
        if time.time() > deadline:
            break
    
    if best_a > best_b:
        best_a, best_b = best_b, best_a
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
