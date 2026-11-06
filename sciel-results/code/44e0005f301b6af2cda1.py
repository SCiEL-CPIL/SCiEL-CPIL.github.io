import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need P(n+1) for n=N, so up to N+1
    
    # 1. Generate primes up to sqrt(MAX_VAL) using simple sieve
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Initialize powerful part array
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # 3. Compute powerful parts using vectorized sieve
    for p in primes:
        p = int(p)
        q = p * p
        while q <= MAX_VAL:
            next_q = q * p
            
            if time.time() > deadline - 3:
                break
            
            # For multiples of q that are NOT multiples of next_q, multiply by q
            # These are indices: q, 2q, ..., max_k*q where k % p != 0
            # i.e., k in {1,2,...,p-1} mod p
            max_k = MAX_VAL // q
            if max_k > 0:
                # Generate indices directly using arange with step
                # For each residue r in 1..p-1 (mod p), take q*r, q*(r+p), q*(r+2p), ...
                # This avoids creating a full array and masking
                for r in range(1, p):
                    start_idx = q * r
                    if start_idx > MAX_VAL:
                        break
                    # All indices: start_idx, start_idx + next_q, start_idx + 2*next_q, ...
                    count = (MAX_VAL - start_idx) // next_q + 1
                    if count > 0:
                        idx = np.arange(start_idx, MAX_VAL + 1, next_q, dtype=np.int64)
                        P[idx] *= q
            
            q = next_q
            
            if time.time() > deadline - 3:
                break
        
        if time.time() > deadline - 3:
            break
    
    # 4. Find best n in [10000, 1000000]
    best_n = 10000
    best_val = int(P[10000]) * int(P[10001])
    
    # Convert to Python list for faster scalar access in the final loop
    P_list = P.tolist()
    
    for n in range(10000, N + 1):
        val = P_list[n] * P_list[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline - 1:
            break
    
    return best_n
