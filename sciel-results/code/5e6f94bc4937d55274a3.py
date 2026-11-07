"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX = 10000000
    
    # Precompute smallest prime factor for fast factorization
    spf = list(range(MAX + 1))
    for i in range(2, isqrt(MAX) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Cache for sigma values
    sigma_cache = {}
    
    def sigma(n):
        """Compute sigma(n) using precomputed SPF."""
        if n in sigma_cache:
            return sigma_cache[n]
        result = 1
        while n > 1:
            p = spf[n]
            # Sum of powers of p
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            result *= power_sum
        sigma_cache[n] = result
        return result
    
    best_err_num = None
    best_err_den = None
    best_a = 11
    best_b = 6
    
    # Strategy: For each b, the minimal possible error is:
    # - If b is odd: 1/(2b) (achieved when a = (3b±1)/2)
    # - If b is even: 1/b (achieved when a = 3b/2 ± 1)
    # 
    # So the best possible error decreases as b increases.
    # We should search large b first to find good candidates early,
    # then use that to prune smaller b.
    #
    # For a given error bound E = num/den, we can prune b if:
    #   1/(2b) >= E  =>  b <= 1/(2E) = den/(2*num)
    # So if we have a candidate with error num/den, we only need to check
    # b > den/(2*num) for potentially better errors.
    
    # Search b from large to small, checking tight windows first
    # For each b, check a candidates in order of increasing error
    
    # Phase 1: Search large b (10^6 to 10^7) with tight window
    # These can achieve error as small as ~1/(2*10^7) = 5e-8
    
    for b in range(MAX, 1000000, -1):
        if time.time() > deadline - 5:
            break
        
        # Pruning: if we already have error num/den, skip b if 1/(2b) >= num/den
        if best_err_num is not None:
            # 1/(2b) >= num/den  =>  den >= 2b*num  =>  b <= den/(2*num)
            if b <= best_err_den // (2 * best_err_num):
                continue
        
        sb = sigma(b)
        
        if b % 2 == 0:
            a_target = (3 * b) // 2
            # Check a_target ± 1, ± 2, ± 3
            for d in range(1, 4):
                for a_cand in (a_target - d, a_target + d):
                    if 1 < a_cand <= MAX:
                        if sigma(a_cand) == sb:
                            num = 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
        else:
            a_lo = (3 * b - 1) // 2
            a_hi = (3 * b + 1) // 2
            # Check a_lo, a_hi, then a_lo-1, a_hi+1, etc.
            for d in range(0, 3):
                for a_cand in (a_lo - d, a_hi + d):
                    if 1 < a_cand <= MAX:
                        if sigma(a_cand) == sb:
                            num = 1 + 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
    
    # Phase 2: Search medium b (100000 to 10^6) with slightly wider window
    for b in range(1000000, 100000, -1):
        if time.time() > deadline - 3:
            break
        
        # Pruning
        if best_err_num is not None:
            if b <= best_err_den // (2 * best_err_num):
                continue
        
        sb = sigma(b)
        
        if b % 2 == 0:
            a_target = (3 * b) // 2
            for d in range(1, 6):
                for a_cand in (a_target - d, a_target + d):
                    if 1 < a_cand <= MAX:
                        if sigma(a_cand) == sb:
                            num = 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
        else:
            a_lo = (3 * b - 1) // 2
            a_hi = (3 * b + 1) // 2
            for d in range(0, 5):
                for a_cand in (a_lo - d, a_hi + d):
                    if 1 < a_cand <= MAX:
                        if sigma(a_cand) == sb:
                            num = 1 + 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
    
    # Phase 3: Search small b (2 to 100000) with wider window
    for b in range(100000, 1, -1):
        if time.time() > deadline - 1:
            break
        
        # Pruning
        if best_err_num is not None:
            if b <= best_err_den // (2 * best_err_num):
                continue
        
        sb = sigma(b)
        
        if b % 2 == 0:
            a_target = (3 * b) // 2
            for d in range(1, 20):
                for a_cand in (a_target - d, a_target + d):
                    if 1 < a_cand <= MAX:
                        if sigma(a_cand) == sb:
                            num = 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
        else:
            a_lo = (3 * b - 1) // 2
            a_hi = (3 * b + 1) // 2
            for d in range(0, 19):
                for a_cand in (a_lo - d, a_hi + d):
                    if 1 < a_cand <= MAX:
                        if sigma(a_cand) == sb:
                            num = 1 + 2 * d
                            den = 2 * b
                            if best_err_num is None or num * best_err_den < best_err_num * den:
                                best_err_num = num
                                best_err_den = den
                                best_a = a_cand
                                best_b = b
    
    if best_err_num is not None:
        return [int(best_a), int(best_b)]
    else:
        return [11, 6]
# EVOLVE-BLOCK-END
