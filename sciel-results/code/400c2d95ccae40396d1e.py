import numpy as np

def construct():
    N = 10000000
    
    # Compute smallest prime factor (SPF) sieve
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma(n) for all n
    sigma = np.ones(N + 1, dtype=np.int64)
    # sigma[1] = 1 is already set by np.ones
    for n in range(2, N + 1):
        if spf[n] == n:
            sigma[n] = n + 1
        else:
            temp = n
            result = 1
            while temp > 1:
                p = spf[temp]
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                result *= (power * p - 1) // (p - 1)
            sigma[n] = result
            
    best_error = float('inf')
    best_pair = (1, 1)
    
    # Iterate k starting from 1
    # Max k such that min error k/(2*1) < best_error? 
    # Actually, we break if min possible error for current k >= best_error.
    # Min possible error for k is k / (2 * b_max) where b_max = floor((2N-k)/3).
    
    for k in range(1, N + 1):
        # Calculate max b for this k
        # 3b + k <= 2N  => b <= (2N - k) / 3
        b_max = (2 * N - k) // 3
        if b_max < 1:
            break
            
        # Check if we can beat current best
        # Min error for this k is k / (2 * b_max)
        min_possible_error = k / (2.0 * b_max)
        if min_possible_error >= best_error:
            break
            
        # Determine starting b and step
        # 2a = 3b + k. 2a is even => 3b + k is even.
        # If k is even, 3b is even => b is even.
        # If k is odd, 3b is odd => b is odd.
        if k % 2 == 0:
            # b must be even
            start_b = b_max if b_max % 2 == 0 else b_max - 1
            step = -2
        else:
            # b must be odd
            start_b = b_max if b_max % 2 == 1 else b_max - 1
            step = -2
            
        if start_b < 1:
            continue
            
        found_for_k = False
        for b in range(start_b, 0, step):
            a = (3 * b + k) // 2
            # a is guaranteed to be <= N and >= 1
            if sigma[a] == sigma[b]:
                error = k / (2.0 * b)
                if error < best_error:
                    best_error = error
                    best_pair = (a, b)
                found_for_k = True
                break # Largest b found for this k is the best for this k
                
    return [best_pair[0], best_pair[1]]
