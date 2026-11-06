import math
import time
import numpy as np

def _powerful_part_direct(n):
    """Compute powerful part of n by direct factorization."""
    if n < 4:
        return 1
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            e = 0
            while temp % d == 0:
                temp //= d
                e += 1
            if e >= 2:
                result *= d ** e
        d += 1 if d == 2 else 2
    if temp > 1:
        pass  # prime with exponent 1
    return result

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize powerful part array P with 1s
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts using vectorized operations
    for p in primes:
        if time.time() - start_time > deadline - 5:
            break
        p = int(p)
        p2 = p * p
        if p2 > MAX_VAL:
            break
        power = p2
        while power <= MAX_VAL:
            if time.time() - start_time > deadline - 5:
                break
            next_power = power * p
            if next_power > MAX_VAL:
                # All multiples of power have exactly this exponent
                P[power::power] *= power
            else:
                # Multiply multiples of power that are NOT multiples of next_power
                # More efficient: use quotient modulo p
                count = (MAX_VAL - power) // power + 1
                multiples = np.arange(power, MAX_VAL + 1, power, dtype=np.int64)
                quotients = multiples // power
                mask = (quotients % p) != 0
                if np.any(mask):
                    valid_indices = multiples[mask]
                    P[valid_indices] *= power
            power = next_power
    
    if time.time() - start_time > deadline - 2:
        return 10000
    
    products = P[10000:1000001] * P[10001:1000002]
    max_idx = np.argmax(products)
    best_n = 10000 + int(max_idx)
    
    # Local refinement: check a window around the best candidate
    window = 100
    lo = max(10000, best_n - window)
    hi = min(1000000, best_n + window)
    best_val = -1
    best_n_final = best_n
    for n in range(lo, hi + 1):
        if time.time() - start_time > deadline - 3:
            break
        val = int(P[n]) * int(P[n + 1])
        if val > best_val:
            best_val = val
            best_n_final = n
    
    # Verify correctness with direct computation for the final answer
    pn_sieve = int(P[best_n_final])
    pn1_sieve = int(P[best_n_final + 1])
    pn_direct = _powerful_part_direct(best_n_final)
    pn1_direct = _powerful_part_direct(best_n_final + 1)
    
    if pn_sieve != pn_direct or pn1_sieve != pn1_direct:
        # Sieve has a bug; do exhaustive direct search in a wider window
        if time.time() - start_time < deadline - 10:
            best_val = 0
            best_n_direct = best_n_final
            wide_lo = max(10000, best_n_final - 1000)
            wide_hi = min(1000000, best_n_final + 1000)
            for n in range(wide_lo, wide_hi + 1):
                if time.time() - start_time > deadline - 5:
                    break
                val = _powerful_part_direct(n) * _powerful_part_direct(n + 1)
                if val > best_val:
                    best_val = val
                    best_n_direct = n
            return best_n_direct
    
    return best_n_final
