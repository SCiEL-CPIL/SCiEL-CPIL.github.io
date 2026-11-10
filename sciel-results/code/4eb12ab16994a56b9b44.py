import time
import sys

MAX_VAL = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _build_spf(n):
    """
    Build smallest prime factor (SPF) array for [0, n] using linear sieve.
    """
    spf = list(range(n + 1))
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma_from_spf(n, spf):
    """
    Compute sigma(n) using SPF array.
    sigma(n) = product over prime factors p^e || n of (p^(e+1)-1)/(p-1).
    """
    if n == 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        power = 0
        sum_p = 1
        while n % p == 0:
            n //= p
            power += 1
            sum_p *= p
            sum_p += 1
        # sum_p is 1 + p + ... + p^power
        result *= sum_p
    return result

def construct():
    start_time = time.time()
    
    # Build SPF
    spf = _build_spf(MAX_VAL)
    
    # Fallback
    best_pair = (11, 6)
    
    # Search for k = 1, 2, ...
    # For each k, search b from MAX_VAL down to 1
    # To avoid infinite loop or excessive time, we limit the search depth or time.
    
    # We can optimize by noting that for a fixed k, if we find a solution, it's optimal for that k.
    # Since error = k / (2b), smaller k is always better than larger k for any b <= MAX_VAL.
    # So we just need to find the first k for which a solution exists.
    # And for that k, the first b (largest) we find is the best.
    
    # However, if we iterate b from MAX_VAL down, we might check many b's.
    # Let's estimate. If sigma fibers are dense, we should find a match quickly.
    
    for k in range(1, 1000): # Limit k to prevent infinite loop if no small k solution exists
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Determine step for b
        # 3b +/- k must be even => b and k must have same parity
        # If k is odd, b must be odd. If k is even, b can be any.
        # Let's just iterate all b, but skip based on parity if k is odd to save half the time.
        
        if k % 2 == 1:
            start_b = MAX_VAL if MAX_VAL % 2 == 1 else MAX_VAL - 1
            step = -2
        else:
            start_b = MAX_VAL
            step = -1
            
        for b in range(start_b, 0, step):
            if time.time() - start_time > TIME_LIMIT - 2:
                return [int(best_pair[0]), int(best_pair[1])]
                
            # Case 1: 2a - 3b = k => 2a = 3b + k => a = (3b + k) / 2
            num1 = 3 * b + k
            if num1 % 2 == 0:
                a1 = num1 // 2
                if 1 <= a1 <= MAX_VAL:
                    if _sigma_from_spf(a1, spf) == _sigma_from_spf(b, spf):
                        best_pair = (a1, b)
                        return [int(a1), int(b)]
                        
            # Case 2: 2a - 3b = -k => 2a = 3b - k => a = (3b - k) / 2
            num2 = 3 * b - k
            if num2 > 0 and num2 % 2 == 0:
                a2 = num2 // 2
                if 1 <= a2 <= MAX_VAL:
                    if _sigma_from_spf(a2, spf) == _sigma_from_spf(b, spf):
                        best_pair = (a2, b)
                        return [int(a2), int(b)]
                        
    # If we didn't find anything in the loop, return fallback
    return [int(best_pair[0]), int(best_pair[1])]
