import time
import math
from collections import deque

def construct():
    deadline = time.time() + 80
    TARGET = 123457
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes_all = sieve(5000)
    
    def generate_smooth_numbers(limit, primes):
        smooth = set([1])
        queue = deque([1])
        while queue:
            x = queue.popleft()
            for p in primes:
                nx = x * p
                if nx > limit:
                    break
                if nx not in smooth:
                    smooth.add(nx)
                    queue.append(nx)
        return smooth
    
    def check_smooth_pair(M):
        if time.time() > deadline - 5:
            return None
        primes_up_to_M = [p for p in primes_all if p <= M]
        smooth_nums = generate_smooth_numbers(TARGET, primes_up_to_M)
        for a in smooth_nums:
            if a <= 0 or a >= TARGET:
                continue
            b = TARGET - a
            if b <= 0:
                continue
            if b in smooth_nums:
                return [a, b]
        return None
    
    # Binary search on M to find minimal smoothness bound
    # First, find a working upper bound by doubling
    lo, hi = 2, 2
    while hi <= 5000:
        if time.time() > deadline - 5:
            break
        result = check_smooth_pair(hi)
        if result is not None:
            break
        lo = hi + 1
        hi = min(hi * 2, 5000)
    
    if hi > 5000:
        hi = 5000
    
    # Binary search for minimal M in [lo, hi]
    best_result = None
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        result = check_smooth_pair(mid)
        if result is not None:
            best_result = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_result is not None:
        return best_result
    
    return [1, TARGET]
