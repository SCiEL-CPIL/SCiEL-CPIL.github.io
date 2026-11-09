"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

MAX_VAL = 10000000

def _build_spf(n):
    """Build smallest prime factor sieve up to n."""
    spf = list(range(n + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(n)) + 1):
        if spf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma_from_spf(n, spf):
    """Compute sigma(n) using precomputed SPF."""
    if n == 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        power_sum = 1
        power = 1
        while n % p == 0:
            n //= p
            power *= p
            power_sum += power
        result *= power_sum
    return result

def construct():
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # with sigma(a) == sigma(b), 1 <= a, b <= 10^7
    # a should be around 1.5 * b
    
    # Strategy: iterate over b, compute sigma(b), then check a = round(1.5*b) and nearby values
    # But that's too slow for all b up to 10^7.
    
    # Better: use the fact that sigma(a) == sigma(b) is a strong constraint.
    # Let's precompute sigma for all numbers up to 10^7 using a sieve-like approach.
    # sigma(n) can be computed in O(n log n) via a divisor sum sieve.
    
    N = MAX_VAL
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Now group indices by sigma value
    from collections import defaultdict
    sigma_to_indices = defaultdict(list)
    for i in range(1, N + 1):
        s = sigma[i]
        sigma_to_indices[s].append(i)
    
    # For each sigma value, we have a list of numbers with that sigma.
    # We want to find pairs (a, b) from the same list that minimize |a/b - 3/2|.
    # For a given list, sort it, then for each element a, binary search for b ≈ a * 2/3.
    
    best_error_num = float('inf')
    best_error_den = 1
    best_a = None
    best_b = None
    
    for s, indices in sigma_to_indices.items():
        if len(indices) < 2:
            continue
        indices.sort()
        for a in indices:
            # We want b such that a/b ≈ 3/2, so b ≈ a * 2/3
            target_b = a * 2.0 / 3.0
            # Binary search in indices for target_b
            lo = 0
            hi = len(indices) - 1
            # Find the closest element to target_b
            while lo < hi:
                mid = (lo + hi) // 2
                if indices[mid] < target_b:
                    lo = mid + 1
                else:
                    hi = mid
            # Check indices[lo] and indices[lo-1] if they exist
            candidates = []
            if lo < len(indices):
                candidates.append(indices[lo])
            if lo > 0:
                candidates.append(indices[lo - 1])
            for b in candidates:
                if b == a:
                    continue
                # Compute exact error |a/b - 3/2| = |2a - 3b| / (2b)
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    continue
                err_den = 2 * b
                # Compare err_num/err_den with best_error_num/best_error_den
                # Use cross multiplication to avoid floating point
                if err_num * best_error_den < best_error_num * err_den:
                    best_error_num = err_num
                    best_error_den = err_den
                    best_a = a
                    best_b = b
    
    if best_a is None:
        return [1, 1]
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
