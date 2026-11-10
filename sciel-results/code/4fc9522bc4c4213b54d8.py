import numpy as np
import time

def construct():
    start_time = time.time()
    N = 10000000
    # Sieve for sigma(n)
    sigma = np.zeros(N+1, dtype=np.int64)
    # We can optimize the sieve slightly
    # sigma[n] = sum of divisors
    # for i in 1..N: for j in 1..N//i: sigma[i*j] += i
    # Vectorized:
    # for i in 1..N: sigma[i::i] += i
    # This is O(N log N)
    
    # To speed up, we can use a prime factorization sieve to compute sigma directly?
    # sigma(n) is multiplicative.
    # Let's stick to the simple additive sieve first, it's robust.
    
    # Time limit is 60s (evaluator) / 90s (kill).
    # Additive sieve in numpy:
    for i in range(1, N + 1):
        if time.time() - start_time > 80:
            break
        sigma[i::i] += i

    best_error = float('inf')
    best_a = 1
    best_b = 1
    
    # Search for pairs
    # We want to minimize |a/b - 1.5|
    # Iterate b downwards to find large b early, which allows smaller errors
    # b from N down to 1
    # Check a near 1.5b
    
    # If best_error is E, we only care about b > 1/(2E)
    # Initial best_error is inf, so we check all b initially?
    # No, we can start with a known good pair to set a bound.
    # Seed: 11, 6. Error = 1/6.
    # Let's set initial bound to 1/6.
    # Then we only need to check b > 1 / (2 * (1/6)) = 3.
    # This is not a tight bound.
    # We can find a better pair quickly.
    
    # Let's just run the loop.
    # To ensure we don't take too long, we can limit the window of a to check.
    # Window size K=10 is likely sufficient.
    
    for b in range(N, 0, -1):
        if time.time() - start_time > 85:
            break
            
        # If current best error is E, min possible error for this b is 1/(2b) (if 2a-3b=1)
        # If 1/(2b) >= best_error, then no a for this b can improve.
        # So if b <= 1/(2*best_error), we can break?
        # Since b is decreasing, if b is small, 1/(2b) is large.
        # We want small error.
        # If 1/(2b) > best_error, then even the best possible error for this b is worse than current best.
        # So we can skip this b.
        # As b decreases, 1/(2b) increases.
        # So once 1/(2b) > best_error, all subsequent b (smaller) will also have 1/(2b') > best_error.
        # So we can break.
        
        if best_error < float('inf'):
            if 1.0 / (2.0 * b) >= best_error:
                break
        
        target_a = 1.5 * b
        a_center = int(round(target_a))
        
        # Check a in [a_center - 10, a_center + 10]
        # Ensure a is within [1, N]
        start_a = max(1, a_center - 10)
        end_a = min(N, a_center + 10)
        
        for a in range(start_a, end_a + 1):
            # Check invalid case: a/b == 1.5 => 2a == 3b
            if 2 * a == 3 * b:
                continue
                
            if sigma[a] == sigma[b]:
                # Calculate error exactly
                # error = |a/b - 3/2| = |2a - 3b| / 2b
                num = abs(2 * a - 3 * b)
                den = 2 * b
                err = num / den
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
    
    return [best_a, best_b]
