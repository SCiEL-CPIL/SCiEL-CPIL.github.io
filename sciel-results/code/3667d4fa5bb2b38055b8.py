import numpy as np
import time

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Compute sigma(n) for all n <= N
    # Using NumPy for speed
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # sigma[n] = sum of divisors
    # Add i to sigma[i], sigma[2i], ...
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        # Fallback to seed if sieve takes too long (unlikely)
        return [11, 6]

    # 2. Search for best pair
    # Iterate b from N down to 1
    # We want to minimize |2a - 3b| / (2b)
    # Let D = |2a - 3b|. Min D is 1.
    # Error E = D / (2b).
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # We can prune if 1/(2*b) > best_error
    # Because D >= 1, so E >= 1/(2*b).
    # If 1/(2*b) > best_error, then for this b and all smaller b, 
    # the minimum possible error is > best_error.
    
    for b in range(N, 0, -1):
        # Pruning check
        if 1.0 / (2.0 * b) > best_error:
            break
            
        if time.time() > deadline:
            break
            
        sb = sigma[b]
        
        # Target a is approx 1.5 * b
        target_a = 1.5 * b
        a_center = int(round(target_a))
        
        # Check neighbors of a_center
        # D = |2a - 3b|
        # We want small D.
        # Check a in [a_center - 10, a_center + 10]
        for da in range(-10, 11):
            a = a_center + da
            if a < 1 or a > N:
                continue
            
            # Check if 2a == 3b (error 0)
            # If so, invalid.
            if 2 * a == 3 * b:
                continue
                
            if sigma[a] == sb:
                D = abs(2 * a - 3 * b)
                # D is at least 1 here
                error = D / (2.0 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                    # We found a better one. 
                    # Can we break inner loop? 
                    # The error depends on D. Smaller D is better.
                    # We are checking da from -10 to 10. 
                    # D is |2(a_center+da) - 3b|.
                    # a_center is closest to 1.5b. So D is minimized near da=0.
                    # We should check da=0, then +/-1, then +/-2...
                    # To be efficient, we can order checks by |da|? 
                    # Or just scan. 20 checks is cheap.
                    
    return [int(best_a), int(best_b)]
