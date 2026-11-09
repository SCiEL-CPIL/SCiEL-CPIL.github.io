"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import isqrt

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using trial division."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    # Handle factor 2
    if temp % 2 == 0:
        p_pow = 1
        while temp % 2 == 0:
            temp //= 2
            p_pow *= 2
        result *= (2 * p_pow - 1)
    # Handle odd factors
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            p_pow = 1
            while temp % f == 0:
                temp //= f
                p_pow *= f
            result *= (f * p_pow - 1) // (f - 1)
        f += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    if n <= 1:
        return []
    factors = []
    temp = n
    if temp % 2 == 0:
        cnt = 0
        while temp % 2 == 0:
            temp //= 2
            cnt += 1
        factors.append((2, cnt))
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            cnt = 0
            while temp % f == 0:
                temp //= f
                cnt += 1
            factors.append((f, cnt))
        f += 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        p_pow = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def _find_k_for_M(M):
    """
    Find all k <= sqrt(M) such that k * sigma(k) = M.
    Strategy: enumerate divisors d of M where d = sigma(k), then k = M/d.
    We need k <= sqrt(M) and k * sigma(k) = M, i.e., sigma(k) = M/k.
    So for each divisor k of M with k <= sqrt(M), check if sigma(k) == M/k.
    """
    sqrtM = isqrt(M)
    results = []
    # Enumerate divisors k of M up to sqrt(M)
    # For each such k, check k * sigma(k) == M
    # This is O(sqrt(M)) in worst case but M <= 10^18 so sqrt(M) <= 10^9
    # That's too slow. Instead, enumerate divisors of M.
    
    factors = _factorize(M)
    divisors = _all_divisors_from_factors(factors)
    
    for k in divisors:
        if k > sqrtM:
            continue
        if k < 1:
            continue
        # Check k * sigma(k) == M
        s = _sigma(k)
        if k * s == M:
            results.append(k)
    
    return results

def _search_collisions(max_k, deadline):
    """
    Search for collisions in k*sigma(k) for k up to max_k.
    Returns dict: M -> list of k values.
    """
    from collections import defaultdict
    collision_map = defaultdict(list)
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            collision_map[M].append(k)
    
    return collision_map

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline to be safe
    
    # Phase 1: Systematic search for k up to some limit
    # We want to find M with many solutions k where k*sigma(k) = M
    # and k <= sqrt(M)
    
    # Start with a moderate search
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Search in chunks to allow deadline checking
    search_limit = 5_000_000
    chunk_size = 100_000
    
    from collections import defaultdict
    collision_map = defaultdict(list)
    
    for k in range(1, search_limit + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            collision_map[M].append(k)
    
    # Find the M with the most solutions where k <= sqrt(M)
    for M, ks_list in collision_map.items():
        if len(ks_list) < best_count:
            continue
        sqrtM = isqrt(M)
        valid_ks = [k for k in ks_list if k <= sqrtM]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_ks = valid_ks
    
    # Phase 2: If we have time, try to extend the search or verify
    if time.time() < deadline - 5:
        # Try a second pass with larger k values
        extended_limit = min(20_000_000, search_limit + 15_000_000)
        for k in range(search_limit + 1, extended_limit + 1):
            if time.time() > deadline:
                break
            s = _sigma(k)
            M = k * s
            if M <= 10**18:
                collision_map[M].append(k)
        
        # Re-evaluate
        for M, ks_list in collision_map.items():
            if len(ks_list) < best_count:
                continue
            sqrtM = isqrt(M)
            valid_ks = [k for k in ks_list if k <= sqrtM]
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M
                best_ks = valid_ks
    
    # Verify the best result
    # Ensure all k values satisfy k * sigma(k) == M and k <= sqrt(M)
    sqrtM = isqrt(best_M)
    verified_ks = []
    for k in best_ks:
        if k <= sqrtM and k * _sigma(k) == best_M:
            verified_ks.append(k)
    
    # Ensure distinct
    verified_ks = list(set(verified_ks))
    
    if len(verified_ks) == 0:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
