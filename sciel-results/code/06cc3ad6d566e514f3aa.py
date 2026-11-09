import math
from itertools import combinations

def is_perfect_square(n):
    if n < 0:
        return False
    x = math.isqrt(n)
    return x * x == n

def get_sf(n):
    """Get square-free part of n"""
    res = 1
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            while n % (i * i) == 0:
                n //= (i * i)
        i += 1
    return n

# We need 210 * prod(n_i) to be a square.
# This is equivalent to prod(sf(n_i)) * 210 being a square.
# Let target_sf = 210.
# We need a subset of numbers from {211, ..., M} such that the product of their square-free parts is a square multiple of 210.
# Actually, it's simpler to just check the square condition directly on the product.
# But product grows huge.
# Better to use the parity vector approach or just check sf products.
# prod(sf(n_i)) must be of the form 210 * k^2.
# Since sf(n_i) are square-free, let P be the set of primes.
# v_p(prod(sf(n_i))) + v_p(210) must be even.

def solve():
    # Search for minimal M
    for M in range(211, 300):
        nums = list(range(211, M + 1))
        sfs = [get_sf(n) for n in nums]
        
        # We need a subset S such that prod(sfs[i] for i in S) * 210 is a perfect square.
        # This is a subset product problem.
        # Given small M, we can try small subset sizes.
        # Size 1: check if any sf(n) * 210 is a square.
        for i in range(len(nums)):
            if is_perfect_square(sfs[i] * 210):
                return [nums[i]]
        
        # Size 2
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if is_perfect_square(sfs[i] * sfs[j] * 210):
                    return [nums[i], nums[j]]
                    
        # Size 3
        # To save time, we can limit search or break if found.
        # 252 was found with size 3.
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    if is_perfect_square(sfs[i] * sfs[j] * sfs[k] * 210):
                        return [nums[i], nums[j], nums[k]]
                        
    return []

# Run the solver to confirm the minimum
# result = solve()
# print(result, max(result))
