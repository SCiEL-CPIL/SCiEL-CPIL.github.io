import math

def construct():
    N = 123457
    
    # Step 1: Compute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is a prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # lpf[1] is 0, which is effectively <= any K >= 2. 
    # But P(1)=1. The problem says P(1)=1.
    # Our lpf[1] is 0. We should treat 0 as 1 or just ensure K>=1.
    # Since a,b are positive integers, a>=1, b>=1.
    # If a=1, P(a)=1.
    # The condition is P(a) <= K.
    # If K=1, only a=1 is allowed.
    # Let's set lpf[1] = 1 for consistency, or just handle it in check.
    # Actually, the sieve starts at 2, so lpf[1] remains 0.
    # We want P(1)=1. So let's set lpf[1] = 1.
    lpf[1] = 1

    # Step 2: Binary search for minimum K
    # K must be at least 1.
    # Max K is N (if a is prime N-1 or similar, but max prime factor is N)
    low = 1
    high = N
    
    # Helper function to check if a valid pair exists for a given K
    def exists_pair(K):
        # We need a in [1, N-1] such that lpf[a] <= K and lpf[N-a] <= K
        # Optimization: 
        # If K is small, few numbers are smooth.
        # We can just loop.
        for a in range(1, N):
            if lpf[a] <= K and lpf[N - a] <= K:
                return True
        return False

    # Binary search
    # We want the smallest K such that exists_pair(K) is True.
    # Note: exists_pair is monotonic. If it's true for K, it's true for K+1.
    
    # We can also optimize the search space.
    # The smallest possible max prime factor.
    
    while low < high:
        mid = (low + high) // 2
        if exists_pair(mid):
            high = mid
        else:
            low = mid + 1
            
    min_K = low
    
    # Step 3: Find the actual pair
    for a in range(1, N):
        if lpf[a] <= min_K and lpf[N - a] <= min_K:
            return [a, N - a]
            
    # Fallback (should not happen)
    return [1, N - 1]
