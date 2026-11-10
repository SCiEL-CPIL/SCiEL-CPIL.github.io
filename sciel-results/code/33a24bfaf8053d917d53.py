"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

MAX_VAL = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _build_spf(n):
    """Build smallest prime factor array up to n."""
    spf = list(range(n + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma(n, spf):
    """Compute sigma(n) using SPF."""
    if n == 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        # Collect power of p
        power = 1
        while n % p == 0:
            n //= p
            power *= p
        # sigma(p^k) = (p^(k+1) - 1)/(p - 1)
        result *= (power * p - 1) // (p - 1)
    return result

def construct():
    start_time = time.time()
    
    # Build SPF table
    spf = _build_spf(MAX_VAL)
    
    best_num = None
    best_den = None
    best_pair = (11, 6)  # fallback: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = 1/6
    
    # Search b from large to small (larger b = smaller error denominator = potentially smaller error)
    # For each b, candidate a values near 3b/2
    # error = |2a - 3b| / (2b)
    # For odd b: |2a - 3b| = 1 (since 3b is odd, 2a is even, diff is odd, min is 1)
    #   error = 1/(2b)
    # For even b: 3b/2 is integer, a = 3b/2 gives error 0 (invalid)
    #   next candidates: a = 3b/2 ± 1, error = 2/(2b) = 1/b
    #
    # Strategy: search b from max_b down to 1
    # Early stop: if best_num/best_den is found, for any b where min_possible_error >= best_error, stop
    #   min possible error for odd b is 1/(2b), for even b is 1/b (since error=0 is invalid)
    #   So overall min error at b is 1/(2b) for odd b
    #   Stop when 1/(2b) >= best_num/best_den => best_den >= 2*b*best_num => b <= best_den/(2*best_num)
    
    max_b = (2 * MAX_VAL) // 3  # a = 3b/2 <= MAX_VAL => b <= 2*MAX_VAL/3
    
    for b in range(max_b, 0, -1):
        # Check time every 50000 iterations
        if b % 50000 == 0:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        # Early termination check
        if best_num is not None:
            # If even the best possible error at this b can't beat our current best, stop
            # Best possible error at b: for odd b it's 1/(2b), for even b it's 1/b
            # Conservative: 1/(2b)
            # 1/(2b) >= best_num/best_den  <=>  best_den >= 2*b*best_num
            if best_den >= 2 * b * best_num:
                break
        
        if b % 2 == 1:
            # Odd b: 3b is odd, a must be integer, 2a even
            # 2a - 3b = ±1 => a = (3b±1)/2
            # a_floor = (3b-1)//2, a_ceil = (3b+1)//2
            a1 = (3 * b - 1) // 2
            a2 = (3 * b + 1) // 2
            candidates = [a1, a2]
        else:
            # Even b: 3b/2 is integer
            # a = 3b/2 gives error 0 (invalid)
            # a = 3b/2 ± 1 gives error 2/(2b) = 1/b
            a_mid = 3 * b // 2
            candidates = [a_mid - 1, a_mid + 1]
        
        sigma_b = _sigma(b, spf)
        
        for a in candidates:
            if a < 1 or a > MAX_VAL:
                continue
            if 2 * a == 3 * b:
                continue  # error = 0, invalid
            
            sigma_a = _sigma(a, spf)
            if sigma_a == sigma_b:
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                
                if best_num is None:
                    best_num = error_num
                    best_den = error_den
                    best_pair = (a, b)
                else:
                    # Compare error_num/error_den < best_num/best_den
                    if error_num * best_den < best_num * error_den:
                        best_num = error_num
                        best_den = error_den
                        best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
