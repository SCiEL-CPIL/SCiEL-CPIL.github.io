import time

def construct():
    N = 123457
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0 # 55 seconds out of 60
    
    # Sieve for largest prime factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Optimization: Only iterate i up to sqrt(N) for the sieve?
    # No, we need to mark multiples of all primes.
    # Standard sieve is O(N log log N).
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                # Check time periodically
                if (j & 0xFFFFF) == 0 and time.time() > deadline:
                    # Fallback to a heuristic or best found so far?
                    # But the sieve is very fast for N=123457.
                    # This check is likely never triggered.
                    pass

    best_val = float('inf')
    best_pair = [1, N - 1]
    
    # Only need to check up to N//2
    for a in range(1, N // 2 + 1):
        b = N - a
        val = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if val < best_val:
            best_val = val
            best_pair = [a, b]
        
        # Early termination if we hit the theoretical minimum?
        # The minimum possible max prime factor is at least 2.
        # But for a sum of odd numbers (123457 is odd), one is even, one is odd.
        # The even number has factor 2.
        # Can we get max(P(a), P(b)) = 2?
        # This would require both a and b to be powers of 2.
        # a = 2^x, b = 2^y.
        # 2^x + 2^y = 123457.
        # 123457 is odd. Sum of two even numbers is even.
        # So one must be 1?
        # P(1)=1.
        # If a=1, b=123456.
        # 123456 is even, not a power of 2.
        # So min is > 2.
        
        # If best_val is small, we can stop?
        # We don't know the exact min without checking all.
        # But if best_val == 3?
        # Powers of 2 and 3?
        # It's hard to guess the lower bound for early termination.
        # The loop is fast enough to finish.
        
        if time.time() > deadline:
            break

    return best_pair
