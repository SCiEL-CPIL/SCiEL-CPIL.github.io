<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) by trial division up to sqrt(n)."""
    if n == 1:
        return 1
    result = 1
    temp = n
    # factor out 2
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        result *= (2**(cnt+1) - 1)
    # odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            result *= (p**(cnt+1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _is_valid_pair(k, M):
    """Check that k*sigma(k) == M."""
    if k < 1:
        return False
    if k * k > M:  # k > sqrt(M)
        return False
    s = _sigma(k)
    return k * s == M

def _verify_solution(M, ks):
    """Verify all k in list are valid distinct solutions for M."""
    seen = set()
    for k in ks:
        if not isinstance(k, int) or isinstance(k, bool):
            return False
        if k in seen:
            return False
        seen.add(k)
        if k < 1 or k * k > M:
            return False
        if k * _sigma(k) != M:
            return False
    return True

def _find_solutions_for_M(M, deadline):
    """Find all k <= sqrt(M) such that k*sigma(k) == M."""
    solutions = []
    sqrt_M = math.isqrt(M)
    # We need to search k from 1 to sqrt_M
    # But that's too slow for large M. Instead, we can use the property:
    # k * sigma(k) = M means sigma(k) = M/k, so k must divide M.
    # So we only need to check divisors of M up to sqrt_M.
    
    # Find all divisors of M up to sqrt_M
    # For M up to 10^18, finding all divisors requires factoring M.
    # But we can just iterate and check divisibility, though that's slow.
    # Better: factor M, generate divisors, filter those <= sqrt_M.
    
    # Factor M using trial division (works for M up to ~10^12 quickly, slower for 10^18)
    # For the bootstrap, we'll generate candidate M values that are likely to have many solutions.
    
    # Actually, for this bootstrap, let's just return the known solution.
    # The real search is in construct().
    return solutions

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # safety margin before 150s kill
    
    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3
    
    # Strategy: Generate candidate M values and find all k such that k*sigma(k)=M.
    # Key insight: k*sigma(k) = M implies k divides M and sigma(k) = M/k.
    # To find many solutions, we want M that has many "sigma-divisors".
    
    # Approach 1: Enumerate k values, compute M = k*sigma(k), and group by M.
    # For k up to some limit, compute M and count how many k give the same M.
    # Then pick the M with the most k's.
    
    # This is efficient because we can enumerate k up to, say, 10^6 or 10^7,
    # compute M for each, and use a dictionary to count occurrences.
    # The winner M will have many k values, and we can verify them.
    
    # How high can we go with k? Each k requires computing sigma(k).
    # sigma via trial division up to sqrt(k). For k up to 10^7, sqrt(k) up to 3162.
    # That's manageable. Let's try k up to 2*10^6 first, see how long it takes.
    
    # Use a sieve-like approach for sigma to be faster.
    # Precompute sigma for all n up to N_max using a sieve.
    
    N_max = 2000000  # start with 2M
    # If time allows, we can increase N_max.
    
    # Sieve for sigma
    sigma_arr = [0] * (N_max + 1)
    for i in range(1, N_max + 1):
        for j in range(i, N_max + 1, i):
            sigma_arr[j] += i
    
    # Now build dict: M -> list of k
    M_to_ks = defaultdict(list)
    for k in range(1, N_max + 1):
        M = k * sigma_arr[k]
        if M <= 10**18:
            M_to_ks[M].append(k)
    
    # Find M with most k's
    for M, ks in M_to_ks.items():
        if len(ks) > best_count:
            # Verify all k's
            if _verify_solution(M, ks):
                best_count = len(ks)
                best_result = {'M': int(M), 'k': [int(x) for x in ks]}
    
    # Check if we have time for a larger N_max
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30 and best_count < 10:
        # Try a larger sieve
        N_max2 = 5000000
        sigma_arr2 = [0] * (N_max2 + 1)
        for i in range(1, N_max2 + 1):
            for j in range(i, N_max2 + 1, i):
                sigma_arr2[j] += i
        
        for k in range(1, N_max2 + 1):
            M = k * sigma_arr2[k]
            if M <= 10**18:
                M_to_ks[M].append(k)
        
        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                if _verify_solution(M, ks):
                    best_count = len(ks)
                    best_result = {'M': int(M), 'k': [int(x) for x in ks]}
    
    return best_result

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from collections import defaultdict
import numpy as np

def _sigma(n):
    """Compute sigma(n) by trial division up to sqrt(n)."""
    if n == 1:
        return 1
    result = 1
    temp = n
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        result *= (2**(cnt+1) - 1)
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            result *= (p**(cnt+1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _verify_solution(M, ks):
    """Verify all k in list are valid distinct solutions for M."""
    seen = set()
    for k in ks:
        if not isinstance(k, int) or isinstance(k, bool):
            return False
        if k in seen:
            return False
        seen.add(k)
        if k < 1 or k * k > M:
            return False
        if k * _sigma(k) != M:
            return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0

    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3

    # Use NumPy vectorized sieve to compute sigma up to large N quickly.
    # sigma(n) = sum of divisors. Sieve: for each d, add d to all multiples.
    # Vectorized: for d in 1..N: sigma[d::d] += d

    # Start with a large N. NumPy vectorized sieve is much faster than pure Python.
    # N=10^7: sieve takes ~O(N log N) additions but vectorized per step.
    # Each step d does N/d additions. Total ~ N * H(N) ~ N * ln(N) ~ 10^7 * 16 ~ 1.6*10^8 ops.
    # NumPy can do ~10^8 simple ops in ~1-2 seconds per large step, but we have N steps.
    # Actually the vectorized approach: for d in range(1, N+1): sigma[d::d] += d
    # The total number of elements touched is N * H(N) ~ N*ln(N).
    # For N=5*10^6: ~5*10^6 * 15.4 ~ 7.7*10^7 element updates. NumPy handles this in ~1-3s.
    # For N=10^7: ~10^7 * 16.1 ~ 1.6*10^8. Maybe ~3-6s.
    # For N=2*10^7: ~2*10^7 * 16.8 ~ 3.4*10^8. Maybe ~8-15s.
    # For N=5*10^7: ~5*10^7 * 17.7 ~ 8.9*10^8. Maybe ~20-40s.

    # Strategy: try increasing N values until time runs out.
    # For each N, compute sigma via sieve, then build M->ks dict, find best.

    # We need to be careful: M = k * sigma(k) must be <= 10^18.
    # For k up to 10^7, sigma(k) up to ~k * (some factor), so M up to ~10^14. Fine.
    # For k up to 10^8, sigma(k) up to ~10^8 * 10 = 10^9, M up to ~10^17. Fine.
    # For k up to 10^9, M could exceed 10^18 for some k. Need to filter.

    # Let's try N values: 5M, 10M, 20M, 50M, 100M
    N_candidates = [5000000, 10000000, 20000000, 50000000, 100000000]

    M_to_ks = defaultdict(list)
    processed_up_to = 0

    for N_max in N_candidates:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break

        # Compute sigma up to N_max using vectorized sieve
        sigma_arr = np.zeros(N_max + 1, dtype=np.int64)
        for d in range(1, N_max + 1):
            sigma_arr[d::d] += d

        # Build M -> ks mapping for k in (processed_up_to, N_max]
        # Actually, let's just process all k from 1 to N_max each time
        # but only add new ones to avoid duplicates. Since we're re-sieving,
        # let's just rebuild from scratch for correctness and simplicity.
        # Better: only process k from processed_up_to+1 to N_max using the new sigma_arr.
        # But sigma_arr is for 1..N_max, so we can use it for any k in range.

        # For efficiency, only add k values not yet processed
        for k in range(processed_up_to + 1, N_max + 1):
            M = k * int(sigma_arr[k])
            if M <= 10**18:
                M_to_ks[M].append(k)

        processed_up_to = N_max

        # Find best M so far
        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                if _verify_solution(M, ks):
                    best_count = len(ks)
                    best_result = {'M': int(M), 'k': [int(x) for x in ks]}

        elapsed = time.time() - start_time
        # If we've already found a good result, we can stop early
        if best_count >= 10 and elapsed > DEADLINE - 30:
            break

    return best_result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
