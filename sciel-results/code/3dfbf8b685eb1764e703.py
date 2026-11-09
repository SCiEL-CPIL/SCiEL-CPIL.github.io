import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin out of 60s limit
    
    N = 1000000
    # Initialize phi array
    phi = np.arange(N + 1)
    
    # Sieve of Euler for totient function
    for i in range(2, N + 1):
        if phi[i] == i:
            # i is prime
            # Update phi for multiples of i
            # phi[j] = phi[j] * (1 - 1/i) = phi[j] - phi[j] // i
            phi[i::i] -= phi[i::i] // i
            
        # Optional: Check time limit
        if time.time() > deadline:
            break
            
    # Find the smallest n > 0 such that phi[n] % 120 == 0
    # phi[0] is 0, phi[1] is 1.
    # We want the first index in [1, N] where condition holds.
    matches = np.where((phi[1:] % 120 == 0))[0]
    
    if len(matches) > 0:
        return int(matches[0] + 1)
    else:
        # Fallback, though there should always be a solution
        return 14400
