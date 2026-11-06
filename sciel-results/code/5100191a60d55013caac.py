import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIME_LIMIT = 80  # Seconds, leave 10s margin for safety
    
    N_MIN = 10000
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    
    # 1. Compute Largest Prime Factor (LPF) for all numbers up to N_MAX + 1
    # Using a sieve-like approach similar to the current program which is efficient.
    # lpf[x] stores the largest prime factor of x.
    lpf = [0] * (N_PLUS_1 + 1)
    
    # Iterate through potential primes
    # If lpf[i] is 0, i is prime.
    # We update all multiples of i. Since we iterate i in increasing order,
    # the last prime i that divides j will be the largest prime factor of j.
    for i in range(2, N_PLUS_1 + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N_PLUS_1 + 1, i):
                lpf[j] = i
    
    # 2. Search for optimal n
    # The score for a candidate n is max(lpf[n], lpf[n+1]).
    # We want to minimize this score.
    
    best_n = N_MIN
    min_score = float('inf')
    
    # Pre-compute scores for all n to allow for more sophisticated search if needed,
    # but a linear scan is O(N) which is fast enough for N=1000000.
    # However, we can optimize the inner loop by avoiding function calls and using local variables.
    
    # Let's do a linear scan first.
    # 1 million iterations in Python takes ~0.1-0.2s.
    # The sieve takes a bit longer but is also fast.
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Largest prime factor of n(n+1) is max(lpf(n), lpf(n+1))
        # because prime factors of n and n+1 are disjoint (gcd(n, n+1)=1)
        lp_n = lpf[n]
        lp_np1 = lpf[n + 1]
        score = lp_n if lp_n > lp_np1 else lp_np1
        
        if score < min_score:
            min_score = score
            best_n = n
            
            # Early termination heuristic?
            # The minimum possible score is 2 (if both are powers of 2, impossible for consecutive > 1).
            # Next is 3? No, 2 and 3 are consecutive. 4 and 5? 5 is prime.
            # 8 and 9? 9=3^2, 8=2^3. Max LPF is 3.
            # So if we find score 3, we can stop?
            # Let's check if 8 and 9 are in range. 8 < 10000.
            # Are there other pairs with max LPF 3?
            # Powers of 2 and 3.
            # 2^a and 3^b.
            # 2^a - 3^b = 1 or 3^b - 2^a = 1.
            # Catalan's conjecture (Mihăilescu's theorem) says the only solution in natural numbers for x^a - y^b = 1 with x, y > 0 and a, b > 1 is 3^2 - 2^3 = 1.
            # So 9 and 8 is the only pair of perfect powers (with exponents > 1) that differ by 1.
            # What about non-perfect powers?
            # We need n and n+1 to be 3-smooth (only factors 2 and 3).
            # 3-smooth numbers: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243, 256, 288, 324, 384, 432, 486, 512, 576, 648, 729, 768, 864, 972, 1024, 1152, 1296, 1458, 1536, 1728, 1944, 2048, 2304, 2592, 2916, 3072, 3456, 3888, 4096, 4608, 5184, 5832, 6144, 6912, 7776, 8192, 9216, 10368, 11664, 12288, 13824, 15552, 16384, 18432, 20736, 23328, 26214, 29491, 32768, 36864, 41472, 46656, 52428, 58982, 65536, 73728, 82944, 92610, 104857, 117964, 131072, 147456, 165888, 185220, 209715, 235929, 262144, 294912, 331776, 370440, 419430, 471858, 524288, 589824, 663552, 740880, 838860, 943716, 1048576, 1179648, 1327104, 1481760, 1677720, 1887432, 2097152, 2359296, 2654208, 2963520, 3355440, 3774864, 4194304, 4718592, 5308416, 5927040, 6710880, 7549728, 8388608, 9437184, 10616832, 11854080, 13421760, 15094368, 16777216, 18874368, 21233664, 23708160, 26843520, 30188736, 33554432, 37748736, 42467328, 47416320, 53687040, 60377472, 67108864, 75497472, 84934656, 94832640, 107374080, 120754944, 134217728, 150994944, 169869312, 189665280, 214748160, 241509888, 268435456, 301989888, 339738624, 379330560, 429496320, 483019776, 536870912, 603979776, 679477248, 758661120, 858992640, 966039552, 1073741824, ...
            # Consecutive 3-smooth numbers:
            # 1, 2
            # 2, 3
            # 3, 4
            # 8, 9
            # Are there any others?
            # It is known that the only pairs of consecutive 3-smooth numbers are (1,2), (2,3), (3,4), (8,9).
            # Proof sketch: If n and n+1 are 3-smooth, then n = 2^a 3^b and n+1 = 2^c 3^d.
            # WLOG a>0 or b>0.
            # If both a,c > 0, then n and n+1 are even, impossible.
            # So one is a power of 2 times a power of 3, and the other is just a power of 2 or power of 3?
            # No, 3-smooth allows both 2 and 3.
            # But they are coprime. So one is of form 2^a and the other 3^b?
            # No, 2^a 3^b and 2^c 3^d. Since gcd=1, one has only 2s and the other only 3s?
            # No. 2^a 3^b and 2^c 3^d.
            # If a>0 and c>0, both even. Impossible.
            # So either a=0 or c=0.
            # Case 1: a=0. n = 3^b. n+1 = 2^c 3^d.
            # Since gcd(3^b, 2^c 3^d)=1, d must be 0.
            # So n = 3^b, n+1 = 2^c.
            # 2^c - 3^b = 1.
            # Solutions: (c=1, b=0) -> 2-1=1. (2,1)
            # (c=2, b=1) -> 4-3=1. (4,3)
            # (c=3, b=2) -> 8-9=-1. (9,8) -> 3^2 - 2^3 = -1.
            # (c=4, b=?) 16-1=15 (not power of 3), 16-9=7.
            # Only solutions to x^a - y^b = 1 are known.
            # Case 2: c=0. n = 2^a 3^b, n+1 = 3^d.
            # gcd(2^a 3^b, 3^d)=1 => b=0.
            # So n = 2^a, n+1 = 3^d.
            # 3^d - 2^a = 1.
            # (d=1, a=0) -> 3-1=2. No.
            # (d=1, a=1) -> 3-2=1. (3,2)
            # (d=2, a=?) 9-1=8=2^3. (9,8)
            # (d=3, a=?) 27-1=26.
            # So the pairs are (1,2), (2,3), (3,4), (8,9).
            # All have max LPF <= 3.
            # Since N_MIN = 10000, none of these are in range.
            # So the minimum score is at least 5.
            
            # Can we get 5?
            # Pairs of 5-smooth numbers.
            # This is more complex.
            # But since we are doing a full scan, we will find the optimal one anyway.
            # The scan is fast enough.
            
            # However, if we find a very small score, we might be able to stop early?
            # No, because we don't know the lower bound.
            # But we can check: if min_score == 5, is it possible to get lower?
            # We established min is >= 5.
            # So if we find 5, we can stop?
            # Wait, is 5 the absolute minimum for n >= 10000?
            # Let's check if there are consecutive 5-smooth numbers >= 10000.
            # If yes, the answer is 5.
            # If no, it's higher.
            # We can just let the scan finish. It's O(N).
            
            # Optimization: If min_score reaches 5, we can break because 5 is the theoretical minimum for n>4.
            # Let's verify this logic.
            # Max LPF of n(n+1).
            # If n, n+1 are 5-smooth, max LPF is 5.
            # If not, max LPF >= 7.
            # So if we find a pair with max LPF 5, it is globally optimal (for n>4).
            # So we can break early if min_score == 5.
            
            if min_score == 5:
                break
                
    return best_n
# EVOLVE-BLOCK-END
