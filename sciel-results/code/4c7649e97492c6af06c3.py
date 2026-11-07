import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_VAL = 10210
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:  # i is prime
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
    
    # Precompute square-free parts for all numbers 211..10210
    sf_cache = {}
    for i in range(211, MAX_VAL + 1):
        sf_cache[i] = compute_sf(i)
    
    # Build reverse map: sf_value -> sorted list of numbers with that sf
    sf_to_nums = defaultdict(list)
    for i in range(211, MAX_VAL + 1):
        sf_to_nums[sf_cache[i]].append(i)
    for v in sf_to_nums:
        sf_to_nums[v].sort()
    
    # Precompute prime factorization of all possible sf values using spf
    # sf values can be up to ~10210 (product of distinct primes)
    # Actually sf values are square-free, so they're products of distinct primes
    # We need to XOR two sf values. Let's use a bitmask approach or direct factorization.
    
    def sf_xor(sf_a, sf_b):
        """Return sf such that sf(sf_a * sf_b) = result (XOR of prime sets)."""
        # Factorize sf_a and sf_b using spf
        def prime_set(n):
            ps = set()
            x = n
            while x > 1:
                p = spf[x]
                ps.add(p)
                while x % p == 0:
                    x //= p
            return ps
        
        pa = prime_set(sf_a)
        pb = prime_set(sf_b)
        pc = pa ^ pb  # symmetric difference
        result = 1
        for p in pc:
            result *= p
        return result
    
    # 1. Single element: [840], raw = 630
    best_sol = [840]
    best_raw = 630
    
    # 2. Pair search
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
    
    # 3. Triplet search: iterate M (max element) upward
    for M in range(211, min(best_raw + 210, MAX_VAL + 1)):
        if time.time() > deadline:
            break
        sf_M = sf_cache[M]
        T = sf_xor(sf_M, 210)
        # Need a, c in (210, M), a < c, sf(a*c) = T
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
                        # Verify
                        prod = a * c * M * 210
                        k = math.isqrt(prod)
                        if k * k == prod:
                            best_raw = M - 210
                            best_sol = [a, c, M]
                            return [int(a), int(c), int(M)]
    
    # 4. Four-element search if needed
    if best_raw > 50:
        # Iterate max element M upward
        for M in range(211, min(best_raw + 210, MAX_VAL + 1)):
            if time.time() > deadline:
                break
            sf_M = sf_cache[M]
            # Need a, b, c in (210, M), all distinct, sf(a*b*c) = complement(sf_M)
            T = sf_xor(sf_M, 210)
            # For each pair (a, b) < M, compute needed sf_c
            # This is O(M^2) per M, too slow. Skip for now.
            pass
    
    result = [int(x) for x in best_sol]
    return result
