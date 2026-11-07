import math
import time
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    MAX_VAL = 10210
    MIN_VAL = 211
    
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
    for i in range(MIN_VAL, MAX_VAL + 1):
        sf_cache[i] = compute_sf(i)
        
    # Collect all unique SF values
    all_sfs = set(sf_cache.values())
    all_sfs.add(210)
    all_sfs.add(1)
    
    # Find all primes up to sqrt(MAX_VAL)
    primes = []
    for i in range(2, int(MAX_VAL ** 0.5) + 2):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # Map prime to index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Map SF to bitmask
    sf_to_mask = {}
    mask_to_sf = {}
    for sf in all_sfs:
        mask = 0
        x = sf
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            # x is a prime factor larger than sqrt(MAX_VAL) or remaining
            # It must be in primes if it's small enough, or we need to handle it
            # Since sf is square-free, x is prime.
            if x in prime_to_idx:
                mask |= (1 << prime_to_idx[x])
            else:
                # This shouldn't happen if primes list is sufficient
                # But just in case, we can't represent it easily.
                # However, sf <= MAX_VAL, so prime factors are <= MAX_VAL.
                # We only need primes up to MAX_VAL for the mask?
                # No, we just need to distinguish the sets.
                # If a prime is > sqrt(MAX_VAL), it appears at most once.
                # We can just add it to the primes list if it appears.
                pass
        
        # Re-derive primes to include all factors of SFs
        # Let's just use a set of primes found in all SFs
        pass
        
    # Better approach: Collect all prime factors of all SFs
    all_primes_in_sfs = set()
    for sf in all_sfs:
        x = sf
        i = 2
        while i * i <= x:
            if x % i == 0:
                all_primes_in_sfs.add(i)
                while x % i == 0:
                    x //= i
            else:
                i += 1
        if x > 1:
            all_primes_in_sfs.add(x)
            
    primes = sorted(all_primes_in_sfs)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    for sf in all_sfs:
        mask = 0
        x = sf
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        sf_to_mask[sf] = mask
        mask_to_sf[mask] = sf
        
    # Build reverse map: sf_value -> sorted list of numbers
    sf_to_nums = defaultdict(list)
    for i in range(MIN_VAL, MAX_VAL + 1):
        sf_to_nums[sf_cache[i]].append(i)
    for v in sf_to_nums:
        sf_to_nums[v].sort()
        
    def sf_xor_mask(sf_a, sf_b):
        """Return sf_c such that sf_a XOR sf_b = sf_c (in prime set sense)."""
        mask_a = sf_to_mask.get(sf_a, 0)
        mask_b = sf_to_mask.get(sf_b, 0)
        mask_c = mask_a ^ mask_b
        return mask_to_sf.get(mask_c, None)
        
    # 1. Single element
    best_sol = [840]
    best_raw = 630
    
    # 2. Pair search
    for a in range(MIN_VAL, MAX_VAL + 1):
        if a >= best_raw + 210:
            break
        if time.time() > deadline:
            break
        sf_a = sf_cache[a]
        sf_b_needed = sf_xor_mask(sf_a, 210)
        if sf_b_needed is not None and sf_b_needed in sf_to_nums:
            nums = sf_to_nums[sf_b_needed]
            idx = bisect.bisect_right(nums, a)
            if idx < len(nums):
                b = nums[idx]
                if b > a and b <= MAX_VAL:
                    raw = b - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_sol = [a, b]
    
    # 3. Triplet search
    for M in range(MIN_VAL, min(best_raw + 210, MAX_VAL + 1)):
        if time.time() > deadline:
            break
        sf_M = sf_cache[M]
        T = sf_xor_mask(sf_M, 210)
        if T is None:
            continue
        for a in range(MIN_VAL, M):
            if time.time() > deadline:
                break
            sf_a = sf_cache[a]
            sf_c_needed = sf_xor_mask(sf_a, T)
            if sf_c_needed is not None and sf_c_needed in sf_to_nums:
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
                            
    # 4. Four-element search
    if best_raw > 100:
        for M in range(MIN_VAL, min(best_raw + 210, MAX_VAL + 1)):
            if time.time() > deadline:
                break
            sf_M = sf_cache[M]
            T = sf_xor_mask(sf_M, 210)
            if T is None:
                continue
            for a in range(MIN_VAL, M):
                if time.time() > deadline:
                    break
                sf_a = sf_cache[a]
                T_ab = sf_xor_mask(sf_a, T)
                if T_ab is None:
                    continue
                for b in range(a + 1, M):
                    if time.time() > deadline:
                        break
                    sf_b = sf_cache[b]
                    sf_c_needed = sf_xor_mask(sf_b, T_ab)
                    if sf_c_needed is not None and sf_c_needed in sf_to_nums:
                        nums_c = sf_to_nums[sf_c_needed]
                        idx = bisect.bisect_right(nums_c, b)
                        if idx < len(nums_c) and nums_c[idx] < M:
                            c = nums_c[idx]
                            if a != c and b != c and a != M and b != M and c != M:
                                prod = a * b * c * M * 210
                                k = math.isqrt(prod)
                                if k * k == prod:
                                    best_raw = M - 210
                                    best_sol = [a, b, c, M]
                                    return [int(a), int(b), int(c), int(M)]
                                    
    result = [int(x) for x in best_sol]
    return result
