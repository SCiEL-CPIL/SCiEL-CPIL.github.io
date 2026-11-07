"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from bisect import bisect_left

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
            result *= (power * p - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    N = 10000000
    candidates = set()
    
    # Add squares
    max_k_sq = int(math.isqrt(N))
    for k in range(1, max_k_sq + 1):
        candidates.add(k * k)
        
    # Add twice-squares
    max_k_twice = int(math.isqrt(N // 2))
    for k in range(1, max_k_twice + 1):
        n = 2 * k * k
        if n <= N:
            candidates.add(n)
            
    # Compute sigma for each candidate and group
    sigma_map = {}
    for n in candidates:
        s = _sigma(n)
        if s in sigma_map:
            sigma_map[s].append(n)
        else:
            sigma_map[s] = [n]
            
    best_err = 0.5
    best_a = 1
    best_b = 1
    
    for s, ns in sigma_map.items():
        if len(ns) < 2:
            continue
            
        # Sort ns to allow binary search
        ns_sorted = sorted(ns)
        
        # For each b in ns_sorted, find a in ns_sorted such that a/b ~ 1.5
        # a ~ 1.5 * b
        for b in ns_sorted:
            target_a = 1.5 * b
            # Find position in ns_sorted
            idx = bisect_left(ns_sorted, target_a)
            
            # Check neighbors
            neighbors = []
            if idx > 0:
                neighbors.append(ns_sorted[idx-1])
            if idx < len(ns_sorted):
                neighbors.append(ns_sorted[idx])
            if idx + 1 < len(ns_sorted):
                neighbors.append(ns_sorted[idx+1])
                
            for a in neighbors:
                # Calculate error |a/b - 1.5|
                # err = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                err = num / den
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
