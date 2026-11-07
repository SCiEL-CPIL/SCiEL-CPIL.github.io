"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n, using trial division up to sqrt(n)."""
    if n == 1:
        return 1
    result = 1
    temp = n
    # Factor out 2
    count = 0
    while temp % 2 == 0:
        count += 1
        temp //= 2
    if count > 0:
        result *= (2**(count+1) - 1)
    # Odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            power = 1
            while temp % p == 0:
                count += 1
                temp //= p
                power *= p
            result *= (power * (p + 1)) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    # Target ratio is 3/2 = 1.5
    # We need sigma(a) == sigma(b) and minimize |a/b - 3/2|
    # a/b should be close to 1.5, so a ≈ 1.5 * b
    # Both a, b <= 10^7
    
    # Strategy: 
    # 1. Note that sigma(n) is odd iff n is a square or twice a square.
    #    But we don't need sigma to be odd, just sigma(a)=sigma(b).
    # 2. Let's try to find pairs where a/b is close to 3/2.
    #    So a = 3k, b = 2k for some k, but sigma(3k) may not equal sigma(2k).
    #    More generally, a/b ≈ 3/2 means a ≈ 3t, b ≈ 2t.
    
    # Better approach: iterate over possible values of b, compute a = round(3*b/2),
    # check if sigma(a) == sigma(b), and track the best.
    
    # But 10^7 iterations with sigma computation might be too slow.
    # Let's be smarter: precompute sigma for a range, or use the fact that
    # we can limit our search.
    
    # Actually, let's try a different approach:
    # Generate candidates where a/b is very close to 3/2.
    # a/b = 3/2 exactly would require 2a = 3b, so a=3k, b=2k.
    # Check if sigma(3k) == sigma(2k) for small k.
    
    # For general search, let's do:
    # For b in range(1, 10^7+1):
    #   a1 = floor(3b/2), a2 = ceil(3b/2)
    #   Check sigma(a1) == sigma(b) and sigma(a2) == sigma(b)
    # This is O(10^7) sigma computations which is too slow.
    
    # Alternative: precompute sigma up to 10^7 using a sieve.
    # sigma sieve: initialize sigma[n] = 1 for all n, then for each d from 1 to 10^7,
    # add d to sigma[d], sigma[2d], sigma[3d], ...
    # This is O(N log N) which for N=10^7 is about 10^7 * ln(10^7) ≈ 10^7 * 16 = 1.6 * 10^8,
    # which might be borderline but let's try.
    
    N = 10000000
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Now search for best pair
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # a/b ≈ 3/2 => a ≈ 1.5 * b
    # For each b, check a = floor(1.5*b) and a = ceil(1.5*b)
    for b in range(1, N + 1):
        # a must be <= N
        a1 = (3 * b) // 2
        a2 = a1 + 1 if a1 * 2 != 3 * b else a1  # if not exact, check both
        
        # Actually, let's just check a1 and a2 if they're different
        candidates = set()
        a1 = (3 * b) // 2
        if a1 >= 1 and a1 <= N:
            candidates.add(a1)
        a2 = a1 + 1
        if a2 >= 1 and a2 <= N:
            candidates.add(a2)
        
        for a in candidates:
            if sigma[a] == sigma[b]:
                # Compute |a/b - 3/2|
                # = |2a - 3b| / (2b)
                err_num = abs(2 * a - 3 * b)
                err_den = 2 * b
                if err_num == 0:
                    # Exact match, but this is invalid per the contract
                    # "A raw value of exactly 0 cannot be scored and counts as invalid"
                    # So we skip exact matches
                    continue
                err = err_num / err_den
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
