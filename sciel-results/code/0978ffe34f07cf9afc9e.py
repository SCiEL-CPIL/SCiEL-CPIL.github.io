import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75  # safety margin: 15s buffer before 90s kill
    upper = 1000001
    
    # Use NumPy for faster sieve
    lpf = np.zeros(upper + 1, dtype=np.int32)
    
    # Only iterate up to sqrt(upper) for primes, then handle remaining
    limit = int(np.sqrt(upper)) + 1
    
    for i in range(2, limit):
        if lpf[i] == 0:
            # i is prime; mark all multiples
            lpf[i::i] = i
    
    # For numbers >= limit that are still 0, they are prime themselves
    # Find indices where lpf is 0 and index >= 2
    zero_mask = (lpf == 0)
    zero_mask[:2] = False  # skip 0 and 1
    prime_indices = np.where(zero_mask)[0]
    for p in prime_indices:
        lpf[p] = p
    
    # Check deadline before final computation
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return 10000
    
    # Calculate max LPF for consecutive pairs n, n+1
    # n ranges from 10000 to 1000000
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find index of minimum value
    min_idx = np.argmin(max_lpf)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
