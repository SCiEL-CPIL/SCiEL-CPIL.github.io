import time
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Generate primes <= 10000
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    N = len(primes)
    
    if N == 0:
        return []
        
    # Max gap between primes <= 10000 is < 100. 200 is safe.
    G_MAX = 200
    
    # L[i][g]: length of longest seq ending at primes[i] with last gap g
    # PM[i][g]: max(L[i][0..g])
    # PM_idx[i][g]: index g' in [0..g] where L[i][g'] == PM[i][g]
    # prev_i[i][g]: previous prime index
    # prev_g[i][g]: previous gap value
    
    # Using lists for speed and simplicity
    L = [[0] * (G_MAX + 1) for _ in range(N)]
    PM = [[0] * (G_MAX + 1) for _ in range(N)]
    PM_idx = [[0] * (G_MAX + 1) for _ in range(N)]
    prev_i = [[0] * (G_MAX + 1) for _ in range(N)]
    prev_g = [[0] * (G_MAX + 1) for _ in range(N)]
    
    best_len = 0
    best_i = 0
    best_g = 0
    
    for i in range(N):
        # Base case: sequence of length 1, gap 0
        L[i][0] = 1
        if 1 > best_len:
            best_len = 1
            best_i = i
            best_g = 0
            
        # Update from all previous primes k < i
        pi = primes[i]
        for k in range(i):
            if time.time() > deadline:
                break
            g = pi - primes[k]
            if g > G_MAX:
                continue
            # Get max length from k with gap <= g
            val = PM[k][g] + 1
            if val > L[i][g]:
                L[i][g] = val
                # Store predecessor info
                # The max in PM[k][g] comes from some g'
                g_prime = PM_idx[k][g]
                prev_i[i][g] = k
                prev_g[i][g] = g_prime
                
                if val > best_len:
                    best_len = val
                    best_i = i
                    best_g = g
                    
        # Compute prefix max and argmax for L[i]
        max_val = 0
        max_idx = 0
        for g in range(G_MAX + 1):
            if L[i][g] > max_val:
                max_val = L[i][g]
                max_idx = g
            PM[i][g] = max_val
            PM_idx[i][g] = max_idx
            
        if time.time() > deadline:
            break

    # Backtrack to reconstruct the sequence
    seq = []
    curr_i = best_i
    curr_g = best_g
    while True:
        seq.append(primes[curr_i])
        if curr_g == 0:
            break
        curr_g = prev_g[curr_i][curr_g]
        curr_i = prev_i[curr_i][curr_g]
        
    seq.reverse()
    
    # Validation
    if not seq:
        return [primes[0]] if N > 0 else []
        
    for j in range(len(seq) - 1):
        if seq[j] >= seq[j+1]:
            # Fallback to first prime
            return [primes[0]]
            
    return seq
