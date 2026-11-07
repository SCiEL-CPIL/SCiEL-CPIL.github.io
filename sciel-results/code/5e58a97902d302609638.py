import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 80
    
    max_val = 10210
    min_val = 211
    
    # Sieve of Eratosthenes to get primes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Map prime to index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_mask(n):
        mask = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                if exp % 2 == 1:
                    if d in prime_to_idx:
                        mask |= (1 << prime_to_idx[d])
            d += 1 if d == 2 else 2
        if temp > 1:
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
        return mask

    # Precompute masks for all n in [min_val, max_val]
    masks = [0] * (max_val + 1)
    for n in range(min_val, max_val + 1):
        masks[n] = get_mask(n)
        
    target_mask = get_mask(210)
    
    # Build map from mask to sorted list of numbers
    mask_to_nums = defaultdict(list)
    for n in range(min_val, max_val + 1):
        mask_to_nums[masks[n]].append(n)
        
    best_max = max_val + 1
    best_set = None
    
    def try_set(s):
        nonlocal best_max, best_set
        m = max(s)
        if m < best_max:
            best_max = m
            best_set = list(s)
    
    # Size 1
    if target_mask in mask_to_nums:
        nums = mask_to_nums[target_mask]
        if nums:
            try_set([nums[0]])
                
    # Size 2
    seen_mask_pairs = set()
    for m in mask_to_nums.keys():
        comp_m = m ^ target_mask
        if comp_m in mask_to_nums:
            pair_key = (min(m, comp_m), max(m, comp_m))
            if pair_key in seen_mask_pairs:
                continue
            seen_mask_pairs.add(pair_key)
            
            nums1 = mask_to_nums[m]
            nums2 = mask_to_nums[comp_m]
            
            if not nums1 or not nums2:
                continue
                
            n1 = nums1[0]
            n2 = nums2[0]
            
            if n1 == n2:
                continue
                
            try_set([n1, n2])
                
    # Size 3
    for x in range(min_val, best_max):
        if time.time() > deadline - 15:
            break
        mx = masks[x]
        for y in range(x + 1, best_max):
            if time.time() > deadline - 15:
                break
            my = masks[y]
            mz = target_mask ^ mx ^ my
            
            if mz in mask_to_nums:
                nums_z = mask_to_nums[mz]
                for candidate in nums_z:
                    if candidate != x and candidate != y:
                        try_set([x, y, candidate])
                        break
    
    # Size 4
    for x in range(min_val, best_max):
        if time.time() > deadline - 5:
            break
        mx = masks[x]
        for y in range(x + 1, best_max):
            if time.time() > deadline - 5:
                break
            if max(x, y) >= best_max:
                break
            my = masks[y]
            for z in range(y + 1, best_max):
                if time.time() > deadline - 5:
                    break
                if max(x, y, z) >= best_max:
                    break
                mz = masks[z]
                mw = target_mask ^ mx ^ my ^ mz
                if mw in mask_to_nums:
                    nums_w = mask_to_nums[mw]
                    for candidate in nums_w:
                        if candidate != x and candidate != y and candidate != z:
                            try_set([x, y, z, candidate])
                            break
    
    if best_set is None:
        best_set = [840]
        best_max = 840
        
    # Verify
    prod = 210
    for x in best_set:
        prod *= x
        
    x_sqrt = math.isqrt(prod)
    if x_sqrt * x_sqrt != prod:
        best_set = [840]
        best_max = 840
        
    return [int(x) for x in best_set]
