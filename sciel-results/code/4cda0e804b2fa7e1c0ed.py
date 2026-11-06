import time
import numpy as np

def _elapsed(start_time):
    return time.time() - start_time

def construct():
    start_time = time.time()
    deadline = 80  # Leave safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Build smallest prime factor sieve up to N_MAX+1
    N = N_MAX + 2
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    if _elapsed(start_time) > deadline - 5:
        return N_MIN
    
    # Precompute powerful part for each number in [2, N_MAX+1]
    # powerful_part(n) = product of p^e for all prime powers p^e || n where e >= 2
    pp = np.ones(N + 1, dtype=np.int64)
    pp[0] = 0
    
    # For each number, compute its powerful part using SPF
    # We can do this efficiently by iterating through numbers and their prime factors
    pp_arr = np.ones(N + 1, dtype=np.int64)
    pp_arr[0] = 0
    pp_arr[1] = 1
    
    for i in range(2, N + 1):
        temp = i
        powerful = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                powerful *= p ** e
        pp_arr[i] = powerful
    
    if _elapsed(start_time) > deadline - 3:
        # Fallback: scan a subset
        best_n = N_MIN
        best_score = pp_arr[N_MIN] * pp_arr[N_MIN + 1]
        for n in range(N_MIN + 1, min(N_MIN + 50000, N_MAX + 1)):
            score = pp_arr[n] * pp_arr[n + 1]
            if score > best_score:
                best_score = score
                best_n = n
        return best_n
    
    # Compute scores = pp[n] * pp[n+1] for n in [N_MIN, N_MAX]
    # Use vectorized multiplication
    scores = pp_arr[N_MIN:N_MAX + 1] * pp_arr[N_MIN + 1:N_MAX + 2]
    
    # Find global maximum
    idx = np.argmax(scores)
    best_n = N_MIN + idx
    best_score = scores[idx]
    
    if _elapsed(start_time) > deadline - 2:
        return best_n
    
    # Local refinement: check around the best point
    search_range = 500
    lo = max(N_MIN, best_n - search_range)
    hi = min(N_MAX, best_n + search_range)
    for n in range(lo, hi + 1):
        score = pp_arr[n] * pp_arr[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
    
    # Also do a broader secondary search in top regions
    # Find top 50 candidates by score and verify
    top_k = min(50, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    for idx2 in top_indices:
        n = N_MIN + idx2
        score = pp_arr[n] * pp_arr[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n
