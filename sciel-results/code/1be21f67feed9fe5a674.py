import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # 1. Precompute largest prime factor for all numbers up to N
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Binary search for the minimum max(P(a), P(b))
    # The value T must be at least 1 and at most N (since P(n) <= n)
    low = 1
    high = N
    best_T = N
    
    def is_valid(T):
        # Check if there exists a in [1, N-1] such that P(a) <= T and P(N-a) <= T
        for a in range(1, N):
            if lpf[a] <= T and lpf[N - a] <= T:
                return True
        return False

    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        if is_valid(mid):
            best_T = mid
            high = mid - 1
        else:
            low = mid + 1
            
    # 3. Find the actual pair (a, b) for the best_T
    # We know a solution exists for best_T
    for a in range(1, N):
        if lpf[a] <= best_T and lpf[N - a] <= best_T:
            return [a, N - a]
            
    # Fallback (should not happen)
    return [1, N - 1]
