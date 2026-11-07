import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10210
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def compute_sf(n):
        sf_val = 1
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                sf_val *= p
        return sf_val
    
    # Precompute square-free parts
    sf_cache = {}
    for i in range(211, MAX_VAL + 1):
        sf_cache[i] = compute_sf(i)
        
    # Build reverse map
    sf_to_nums = defaultdict(list)
    for i in range(211, MAX_VAL + 1):
        sf_to_nums[sf_cache[i]].append(i)
    for v in sf_to_nums:
        sf_to_nums[v].sort()
        
    def sf_xor(sf_a, target):
        def prime_set(n):
            ps = set()
            x = n
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if p * p > x:
                    break
                if x % p == 0:
                    ps.add(p)
                    while x % p == 0:
                        x //= p
            if x > 1:
                ps.add(x)
            return ps
        
        pa = prime_set(sf_a)
        pt = prime_set(target)
        pc = pa ^ pt
        result = 1
        for p in pc:
            result *= p
        return result

    best_sol = [840]
    best_raw = 630
    
    # 1. Pair search
    for a in range(211, MAX_VAL + 1):
        if a >= best_raw + 210:
            break
        if time.time() > deadline:
            break
        sf_a = sf_cache[a]
        sf_b_needed = sf_xor(sf_a, 210)
        if sf_b_needed in sf_to_nums:
            nums = sf_to_nums[sf_b_needed]
            idx = bisect.bisect_right(nums, a)
            if idx < len(nums):
                b = nums[idx]
                if b > a and b <= MAX_VAL:
                    raw = b - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_sol = [a, b]
                        
    # 2. Triplet search
    # Iterate M upward from 211 to best_raw + 210
    for M in range(211, min(best_raw + 210, MAX_VAL + 1)):
        if time.time() > deadline:
            break
        sf_M = sf_cache[M]
        T = sf_xor(sf_M, 210)
        for a in range(211, M):
            if time.time() > deadline:
                break
            sf_a = sf_cache[a]
            sf_c_needed = sf_xor(sf_a, T)
            if sf_c_needed in sf_to_nums:
                nums_c = sf_to_nums[sf_c_needed]
                idx = bisect.bisect_right(nums_c, a)
                if idx < len(nums_c) and nums_c[idx] < M:
                    c = nums_c[idx]
                    if a != c and a != M and c != M:
                        prod = a * c * M * 210
                        k = math.isqrt(prod)
                        if k * k == prod:
                            best_raw = M - 210
                            best_sol = [a, c, M]
                            return [int(a), int(c), int(M)]
                            
    result = [int(x) for x in best_sol]
    return result
