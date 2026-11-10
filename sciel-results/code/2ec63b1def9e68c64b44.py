"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def build_spf(limit):
    """Builds Smallest Prime Factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    limit_sqrt = int(math.isqrt(limit))
    for i in range(2, limit_sqrt + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def compute_powerful_parts(spf, start, end):
    """
    Computes the powerful part for all n in [start, end].
    powerful_part(n) = product of p^e for p^e || n where e >= 2.
    """
    pp = [1] * (end + 2)  # +2 to safely access n+1 if n=end
    spf_local = spf
    
    for n in range(start, end + 1):
        x = n
        val = 1
        while x > 1:
            p = spf_local[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
        
    return pp

def find_best_pair(pp, start, end):
    """
    Finds n in [start, end-1] maximizing pp[n] * pp[n+1].
    """
    best_n = start
    best_val = 0
    # n must be such that n+1 <= end
    # So n goes up to end - 1
    for n in range(start, end):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
    return best_n

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # We need pp[n] and pp[n+1] for n up to MAX_N.
    # So we need to compute pp up to MAX_N + 1.
    LIMIT = MAX_N + 1
    
    # 1. Build SPF
    spf = build_spf(LIMIT)
    
    # Check time
    if time.time() > deadline - 2:
        return MIN_N
        
    # 2. Compute Powerful Parts
    # We only care about n in [MIN_N, MAX_N].
    # But factorization of n depends on primes up to n.
    # The loop in compute_powerful_parts handles this.
    # We compute from MIN_N to LIMIT to cover all n and n+1.
    pp = compute_powerful_parts(spf, MIN_N, LIMIT)
    
    # Check time
    if time.time() > deadline - 2:
        # If we ran out of time, pp might be partially filled with 1s.
        # But compute_powerful_parts doesn't have a time check inside.
        # Given the speed, this shouldn't happen.
        # If it does, we should probably return best found in the computed part.
        # But we don't know which part is computed.
        # Let's assume it finished.
        pass
        
    # 3. Find Best Pair
    # n ranges from MIN_N to MAX_N - 1
    # So n+1 ranges from MIN_N + 1 to MAX_N
    # All indices are within [MIN_N, LIMIT]
    best_n = find_best_pair(pp, MIN_N, MAX_N)
    
    return best_n

# EVOLVE-BLOCK-END
