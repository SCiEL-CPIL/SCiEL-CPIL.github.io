"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

MAX_N = 10_000_000
TARGET_NUM = 3
TARGET_DEN = 2

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Step 1: Build smallest-prime-factor sieve
    spf = list(range(MAX_N + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Step 2: Compute sigma(n) for all n using SPF factorization
    # sigma is multiplicative: sigma(p^k) = (p^(k+1)-1)/(p-1)
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        x = n
        s = 1
        while x > 1:
            p = spf[x]
            pk = 1
            while x % p == 0:
                x //= p
                pk *= p
            s *= (pk * p - 1) // (p - 1)
        sigma[n] = s

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return [11, 6]

    # Step 3: Group numbers by sigma value
    # Use a dict mapping sigma -> list of numbers
    groups = {}
    for n in range(2, MAX_N + 1):
        sv = sigma[n]
        if sv not in groups:
            groups[sv] = []
        groups[sv].append(n)

    # Also include n=1
    if 1 not in groups.get(1, []):
        if 1 not in groups:
            groups[1] = []
        groups[1].append(1)

    # Step 4: For each group, find pair (a, b) minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this. For a given b, best a is near 1.5*b.
    # Sort each group, then for each b, binary search for a near 1.5*b.
    
    best_error_num = None
    best_error_den = None
    best_a = 11
    best_b = 6

    for sv, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() - start_time > DEADLINE:
            break
        nums.sort()
        nlen = len(nums)
        
        for i in range(nlen):
            b = nums[i]
            target_a = 1.5 * b
            # Binary search for position where target_a would be
            lo, hi = 0, nlen - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1 (and maybe lo+1)
            candidates = []
            for pos in [lo - 1, lo, lo + 1]:
                if 0 <= pos < nlen:
                    candidates.append(nums[pos])
            
            for a in candidates:
                if a == b:
                    continue
                # error = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # invalid per task
                den = 2 * b
                # Compare num/den with best_error_num/best_error_den
                if best_error_num is None:
                    best_error_num = num
                    best_error_den = den
                    best_a = a
                    best_b = b
                else:
                    # num/den < best_num/best_den  <=>  num*best_den < best_num*den
                    if num * best_error_den < best_error_num * den:
                        best_error_num = num
                        best_error_den = den
                        best_a = a
                        best_b = b

    # Ensure types are Python int
    best_a = int(best_a)
    best_b = int(best_b)
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
