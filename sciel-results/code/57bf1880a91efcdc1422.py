import math
import time

def _sqfree_kernel(n):
    """Compute the square-free kernel of n (product of primes with odd exponent)."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        count = 0
        while temp % d == 0:
            count += 1
            temp //= d
        if count % 2 == 1:
            result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def construct():
    """
    Returns a list of distinct integers strictly larger than 210 and at most 10210
    such that 210 * product(list) is a perfect square.
    Minimizes max(list) - 210.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    # Known solution: [224, 240] with value 30
    # Search for better solution
    
    # Precompute square-free kernels
    kernels = {}
    for n in range(211, 10211):
        kernels[n] = _sqfree_kernel(n)
    
    # Target kernel is 210 (since 210 = 2*3*5*7, all exponents are 1)
    target = 210
    
    # Search for minimal M: check if we can form target using numbers in (210, M]
    # Try small subsets: 1, 2, 3 elements
    
    # First, check if any single number has kernel 210
    for n in range(211, 10211):
        if time.time() > deadline:
            break
        if kernels[n] == target:
            prod = 210 * n
            s = math.isqrt(prod)
            if s * s == prod:
                return [n]
    
    if time.time() > deadline:
        return [224, 240]
    
    # Check pairs: find minimal max such that some pair (i,j) with i<j<=M has kernel_i ^ kernel_j = 210
    # For each M from 212 upward, check all pairs in (210, M]
    
    # More efficient: for each pair, compute their max, keep track of minimal max
    best_max = 10210
    best_pair = None
    
    for i in range(211, 10211):
        if time.time() > deadline:
            break
        for j in range(i + 1, 10211):
            if time.time() > deadline:
                break
            if (kernels[i] ^ kernels[j]) == target:
                m = max(i, j)
                if m < best_max:
                    best_max = m
                    best_pair = (i, j)
    
    if best_pair is not None:
        i, j = best_pair
        prod = 210 * i * j
        s = math.isqrt(prod)
        if s * s == prod:
            return [i, j]
    
    if time.time() > deadline:
        return [224, 240]
    
    # Check triples
    best_max_triple = 10210
    best_triple = None
    
    # This is O(n^3) which is too slow for n=10000
    # Instead, for each M from 213 upward, check if any triple in (210, M] works
    # But this is also expensive. Let's limit the search.
    
    # Since we already have a solution with max=240, only search M < 240
    for M in range(213, 240):
        if time.time() > deadline:
            break
        nums = list(range(211, M + 1))
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (kernels[nums[i]] ^ kernels[nums[j]] ^ kernels[nums[k]]) == target:
                        prod = 210 * nums[i] * nums[j] * nums[k]
                        s = math.isqrt(prod)
                        if s * s == prod:
                            return [nums[i], nums[j], nums[k]]
    
    # Return the known best solution
    result = [224, 240]
    prod = 210 * 224 * 240
    s = math.isqrt(prod)
    if s * s == prod:
        return result
    
    # Fallback
    return [840]
