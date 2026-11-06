import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    TARGET = 123457
    
    # Sieve primes up to TARGET
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    all_primes = sieve(TARGET)
    
    def generate_smooth_numbers(limit, M):
        # Get primes <= M
        smooth_primes = []
        for p in all_primes:
            if p <= M:
                smooth_primes.append(p)
            else:
                break
        
        if not smooth_primes:
            return {1}
            
        smooth = set([1])
        queue = deque([1])
        
        while queue:
            if time.time() > deadline:
                return smooth # Return partial or handle timeout
            x = queue.popleft()
            for p in smooth_primes:
                nx = x * p
                if nx > limit:
                    break
                if nx not in smooth:
                    smooth.add(nx)
                    queue.append(nx)
        return smooth

    def find_pair(M):
        smooth_nums = generate_smooth_numbers(TARGET, M)
        for a in smooth_nums:
            b = TARGET - a
            if b >= 1 and b in smooth_nums:
                return (a, b)
        return None

    # Binary search for minimal M
    lo = 2
    hi = TARGET
    best_pair = None
    
    # We know a solution exists for M=TARGET (e.g., 1 and TARGET-1)
    # But we want the minimum.
    
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        pair = find_pair(mid)
        if pair is not None:
            best_pair = pair
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_pair is None:
        # Fallback: try linear search from 2 if binary search failed or timed out
        for M in range(2, 100):
            if time.time() > deadline - 2:
                break
            pair = find_pair(M)
            if pair is not None:
                best_pair = pair
                break
                
    if best_pair is None:
        # Absolute fallback
        return [1, TARGET]
        
    return [best_pair[0], best_pair[1]]
