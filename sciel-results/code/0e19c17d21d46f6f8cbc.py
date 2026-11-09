"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    from collections import deque
    
    def mask_of(n):
        m = 0
        for bit, p in enumerate([2, 3, 5, 7]):
            cnt = 0
            while n % p == 0:
                cnt += 1
                n //= p
            if cnt % 2 == 1:
                m |= (1 << bit)
        return m
    
    for M in range(211, 10211):
        pairs = []
        for n in range(211, M + 1):
            pairs.append((n, mask_of(n)))
        
        best_nums = {}
        q = deque()
        q.append((0, ()))
        best_nums[0] = ()
        
        found = False
        while q:
            s, nums = q.popleft()
            if s == 15 and len(nums) > 0:
                nums_list = list(nums)
                prod = 210
                for n in nums_list:
                    prod *= n
                r = isqrt(prod)
                if r * r == prod and all(210 < n <= M for n in nums_list) and len(set(nums_list)) == len(nums_list) and 1 <= len(nums_list) <= 100:
                    return nums_list
                found = True
            if len(nums) >= 4:
                continue
            for n, m in pairs:
                if n in nums:
                    continue
                ns = s ^ m
                if ns not in best_nums or len(nums) < len(best_nums[ns]):
                    best_nums[ns] = nums + (n,)
                    q.append((ns, nums + (n,)))
        
    return [224, 240]
# EVOLVE-BLOCK-END
