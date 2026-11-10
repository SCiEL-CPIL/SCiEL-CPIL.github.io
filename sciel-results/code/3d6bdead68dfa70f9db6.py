import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 10000000

    # Linear sieve for sigma (multiplicative function)
    sigma = np.zeros(N + 1, dtype=np.int64)
    sp = np.zeros(N + 1, dtype=np.int64)  # smallest prime factor
    pe = np.zeros(N + 1, dtype=np.int64)  # highest power of smallest prime dividing n
    sigma[1] = 1
    primes = []

    for i in range(2, N + 1):
        if time.time() - start_time > 20.0:
            break
        if sp[i] == 0:
            sp[i] = i
            pe[i] = i
            sigma[i] = 1 + i
            primes.append(i)
        for p in primes:
            if p > sp[i] or i * p > N:
                break
            sp[i * p] = p
            if i % p == 0:
                pe[i * p] = pe[i] * p
                m = i // pe[i]
                sigma[i * p] = sigma[i] + sigma[m] * pe[i] * p
                break
            else:
                pe[i * p] = p
                sigma[i * p] = sigma[i] * (1 + p)

    if time.time() - start_time > DEADLINE:
        return [int(11), int(6)]

    # Group indices by sigma value
    sigma_to_indices = defaultdict(list)
    for i in range(1, N + 1):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        sigma_to_indices[sigma[i]].append(i)

    best_error = float('inf')
    best_a = 1
    best_b = 1

    # Initial bound from known pair: sigma(11)=12=sigma(6), error=|11/6-3/2|=1/6
    if N >= 11 and sigma[11] == sigma[6]:
        best_error = 1.0 / 6.0
        best_a, best_b = 11, 6

    # For each group with size >= 2, find the pair minimizing |2a - 3b|
    for s, indices in sigma_to_indices.items():
        if time.time() - start_time > DEADLINE - 2.0:
            break
        if len(indices) < 2:
            continue
        
        indices = np.array(indices)
        # Sort indices
        indices.sort()
        
        # For each b in indices, find a in indices closest to 1.5*b
        # We can use binary search
        for b in indices:
            target = 1.5 * b
            # Binary search for target in indices
            pos = np.searchsorted(indices, target)
            
            # Check pos and pos-1
            if pos < len(indices):
                a = indices[pos]
                if 2 * a != 3 * b:
                    err = abs(2 * a - 3 * b) / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = int(a)
                        best_b = int(b)
            if pos > 0:
                a = indices[pos - 1]
                if 2 * a != 3 * b:
                    err = abs(2 * a - 3 * b) / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = int(a)
                        best_b = int(b)

    return [int(best_a), int(best_b)]
