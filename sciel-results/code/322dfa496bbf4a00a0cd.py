"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

MAX_VAL = 10000000

def _build_spf(n):
    """Build smallest prime factor sieve up to n."""
    spf = list(range(n + 1))
    if n >= 0:
        spf[0] = 0
    if n >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(n)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
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
    N = MAX_VAL
    spf = _build_spf(N)
    
    best_error_num = None
    best_error_den = 1
    best_a = None
    best_b = None
    
    # Iterate b from high to low (larger b gives smaller potential error 1/(2b))
    # For each b, check a candidates near 1.5*b
    for b in range(N, 0, -1):
        sigma_b = _sigma_from_spf(b, spf)
        
        # Target a ≈ 1.5 * b
        target_a = (3 * b) // 2
        # Check a few candidates around target_a
        for da in range(-3, 4):
            a = target_a + da
            if a < 1 or a > N:
                continue
            if a == b:
                continue
            # Quick check: error |2a - 3b| / (2b)
            err_num = abs(2 * a - 3 * b)
            if err_num == 0:
                continue
            
            # If we already have a better or equal error with larger b, skip
            # Since err_den = 2*b, and we're going from high b to low b,
            # any solution found later has smaller b, so err_den is smaller
            # We can prune if err_num / (2*b) >= current best
            if best_error_num is not None:
                # Compare err_num/(2*b) with best_error_num/best_error_den
                # i.e., err_num * best_error_den vs best_error_num * 2*b
                if err_num * best_error_den >= best_error_num * (2 * b):
                    continue
            
            sigma_a = _sigma_from_spf(a, spf)
            if sigma_a != sigma_b:
                continue
            
            err_den = 2 * b
            if best_error_num is None or err_num * best_error_den < best_error_num * err_den:
                best_error_num = err_num
                best_error_den = err_den
                best_a = a
                best_b = b
    
    if best_a is None:
        return [1, 1]
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
