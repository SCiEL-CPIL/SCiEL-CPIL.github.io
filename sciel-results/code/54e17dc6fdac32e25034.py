"""Smooth two-summand decomposition of N=123457 minimizing max(P(a),P(b))."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """Return [a, b] with a+b=123457 minimizing max(largest_prime_factor(a), largest_prime_factor(b))."""
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N = 123457
    
    # Strategy: Generate 5-smooth numbers up to N, then scan for pairs where both are 5-smooth.
    # If found, max(P(a), P(b)) <= 5, which is optimal (since P(1)=1, P(2)=2, P(3)=3, P(5)=5).
    # Actually, we want to minimize the max prime factor. Let's think about what's achievable.
    # 
    # First, let's try to find a pair where both a and b have only small prime factors.
    # We'll generate all numbers up to N that are 5-smooth (only primes 2,3,5),
    # then check if any pair sums to N.
    
    # Generate 5-smooth numbers up to N
    smooth_5 = set()
    def gen_smooth(primes, max_val, idx=0, current=1):
        if current > max_val:
            return
        if idx == len(primes):
            smooth_5.add(current)
            return
        for i in range(idx, len(primes)):
            gen_smooth(primes, max_val, i + 1, current)
            if current > max_val // primes[i]:
                return
            current *= primes[i]
    
    gen_smooth([2, 3, 5], N)
    
    # Check for pairs
    best_pair = None
    best_score = float('inf')
    
    for a in smooth_5:
        b = N - a
        if b > 0 and b in smooth_5:
            # Both are 5-smooth, so max(P(a), P(b)) <= 5
            # P(1)=1, so if a=1 or b=1, that's even better
            pa = 1 if a == 1 else 5
            pb = 1 if b == 1 else 5
            score = max(pa, pb)
            if score < best_score:
                best_score = score
                best_pair = (a, b)
    
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
    
    # If no 5-smooth pair found, try 7-smooth numbers
    smooth_7 = set()
    def gen_smooth_7(primes, max_val, idx=0, current=1):
        if current > max_val:
            return
        if idx == len(primes):
            smooth_7.add(current)
            return
        for i in range(idx, len(primes)):
            gen_smooth_7(primes, max_val, i + 1, current)
            if current > max_val // primes[i]:
                return
            current *= primes[i]
    
    gen_smooth_7([2, 3, 5, 7], N)
    
    for a in smooth_7:
        b = N - a
        if b > 0 and b in smooth_7:
            # Both are 7-smooth
            def largest_prime_factor_7smooth(n):
                if n == 1:
                    return 1
                for p in [7, 5, 3, 2]:
                    while n % p == 0:
                        n //= p
                return p if n > 1 else 1  # This is approximate; let's be more careful
            # Actually for 7-smooth, the largest prime factor is at most 7
            pa = 1 if a == 1 else 7
            pb = 1 if b == 1 else 7
            score = max(pa, pb)
            if score < best_score:
                best_score = score
                best_pair = (a, b)
    
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
    
    # If still not found, do a more general search using sieve for largest prime factors
    # Precompute smallest prime factor up to N
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_p = 1
        while n > 1:
            p = spf[n]
            max_p = max(max_p, p)
            while n % p == 0:
                n //= p
        return max_p
    
    # Now scan for the best pair
    # To speed up, we can sort by some heuristic, but let's just scan all a from 1 to N-1
    # and track the best max(P(a), P(b))
    
    # Optimization: we only need to check a up to N//2 due to symmetry
    best_score = float('inf')
    best_a = 1
    
    # Check if we're running out of time
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        score = max(pa, pb)
        if score < best_score:
            best_score = score
            best_a = a
            if best_score <= 5:
                break  # Can't do better than 5 for non-1 values
    
    return [best_a, N - best_a]

# EVOLVE-BLOCK-END
